from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pygame
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Model.policy import ClashRLModel
from clasher.engine import BattleEngine
from clasher.path import DECKS_FILE
from clasher.rl.action_space import DiscreteTileActionSpace
from clasher.rl.deck_pool import apply_deck_to_player, load_deck_pool, sample_decks
from clasher.rl.obs_cv import ObservationBuilder
from clasher.rl.train_selfplay import resolve_torch_device
from clasher.visualizer import (
    ARENA_HEIGHT,
    ARENA_ROWS,
    ARENA_WIDTH,
    ARENA_X,
    ARENA_Y,
    BLUE,
    ELIXIR_PURPLE,
    ELIXIR_PURPLE_LIGHT,
    TILE_SIZE,
    UI_BG,
    UI_BG_LIGHT,
    UI_BORDER,
    UI_GOLD,
    UI_GOLD_DARK,
    UI_TEXT,
    UI_TEXT_DIM,
    BattleVisualizer,
)


NO_OP = 2304
HAND_SIZE = 4
BOARD_WIDTH = 18
BOARD_HEIGHT = 32
TILE_COUNT = BOARD_WIDTH * BOARD_HEIGHT

PANEL_X = ARENA_X + ARENA_WIDTH + 20
PANEL_Y = ARENA_Y
PANEL_WIDTH = 400
PANEL_HEIGHT = ARENA_HEIGHT


def getCheckpointUpdate(path: Path) -> int:
    try:
        match = re.search(r"policy_update_(\d+)", path.stem)
        if match:
            return int(match.group(1))
    except Exception:
        pass
    return 0


def getLatestCheckpoint(directory: Path) -> tuple[Path | None, int]:
    if not directory.exists():
        return None, 0

    checkpoints = []

    for path in directory.glob("policy_update_*.pt"):
        match = re.search(r"policy_update_(\d+)", path.stem)
        if match is None:
            continue
        checkpoints.append((int(match.group(1)), path))

    if not checkpoints:
        return None, 0

    checkpoints.sort(key=lambda item: (item[0], 1 if item[1].stem == f"policy_update_{item[0]}" else 0))
    update, path = checkpoints[-1]
    return path, update


def loadCheckpoint(path: Path, device: torch.device, numThinkSteps: int) -> ClashRLModel:
    state = torch.load(path, map_location=device)
    model = ClashRLModel(num_think_steps=numThinkSteps).to(device)
    model.load_state_dict(state["model_state_dict"])
    return model


def loadTrainingCheckpoint(
    path: Path,
    model: ClashRLModel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state_dict"])

    optimizerState = state.get("optimizer_state_dict")
    if optimizerState is not None:
        optimizer.load_state_dict(optimizerState)

    return int(state.get("update", getCheckpointUpdate(path)))


def saveCheckpoint(
    model: ClashRLModel,
    optimizer: torch.optim.Optimizer,
    outputPath: Path,
    update: int,
    numThinkSteps: int,
) -> None:
    outputPath.parent.mkdir(parents=True, exist_ok=True)
    temporaryPath = outputPath.with_suffix(outputPath.suffix + ".tmp")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "update": update,
            "num_think_steps": numThinkSteps,
        },
        temporaryPath,
    )

    temporaryPath.replace(outputPath)


class HandTrainer(BattleVisualizer):
    def __init__(
        self,
        checkpointPath: Path,
        outputDir: Path,
        opponentCheckpointPath: Path,
        decksPath: Path,
        decisionInterval: int,
        decisionDelayMs: int,
        device: torch.device,
        deterministic: bool,
        opponentDeterministic: bool,
        seed: int | None,
        learningRate: float,
        saveInterval: int,
        continueTraining: bool,
        numThinkSteps: int,
    ):
        self.device = device
        self.checkpointPath = checkpointPath
        self.outputDir = outputDir
        self.opponentCheckpointPath = opponentCheckpointPath
        self.decksPath = decksPath
        self.decisionInterval = decisionInterval
        self.decisionDelayMs = max(0, decisionDelayMs)
        self.deterministic = deterministic
        self.opponentDeterministic = opponentDeterministic
        self.seed = seed
        self.saveInterval = saveInterval
        self.continueTraining = continueTraining
        self.numThinkSteps = numThinkSteps
        self.autoNoOp = False

        self.model = loadCheckpoint(checkpointPath, device, numThinkSteps)
        self.model.train()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learningRate)
        self.currentUpdate = getCheckpointUpdate(checkpointPath)

        if continueTraining:
            self.currentUpdate = loadTrainingCheckpoint(
                checkpointPath,
                self.model,
                self.optimizer,
                device,
            )

        self.nextUpdate = self.currentUpdate + 1

        self.opponentModel = loadCheckpoint(
            opponentCheckpointPath,
            device,
            numThinkSteps,
        )
        self.opponentModel.eval()

        for parameter in self.opponentModel.parameters():
            parameter.requires_grad_(False)

        self.observationBuilder = ObservationBuilder(
            card_vocab=None,
            canonical_perspective=True,
        )
        self.actionSpace = DiscreteTileActionSpace(canonical_perspective=True)
        self.decks = load_deck_pool(decksPath)
        self.rng = np.random.default_rng(seed)

        self.hiddenState = self.model.init_hidden_state(1, device)
        self.opponentHiddenState = self.opponentModel.init_hidden_state(1, device)

        self.lastDecisionTick = 0
        self.decisionCount = 0
        self.lastLoss = None

        self.pendingAction = None
        self.pendingLogits = None
        self.pendingMask = None

        self.scheduledAction = None
        self.scheduledActionAt = None

        self.noopUntilActive = False
        self.noopConditions = []

        self.pausedForDecision = False
        self.correctionMode = False
        self.conditionMenu = False
        self.correctionSlot = None

        self.gameOverMenu = False
        self.quitMenu = False

        self.lastTroopIds = set()
        self.statusText = ""
        self.modelActionText = ""
        self.modelMarkerAction = None
        self.currentDecks = {}

        self.clock = pygame.time.Clock()

        super().__init__()

        self.trainerFont = pygame.font.Font(None, 28)
        self.trainerSmallFont = pygame.font.Font(None, 21)
        self.trainerLargeFont = pygame.font.Font(None, 34)

        self.setupBattle()

    def getNextCheckpointPath(self):
        path = self.outputDir / f"policy_update_{self.nextUpdate}.pt"
        self.nextUpdate += 1
        return path

    def saveTrainingCheckpoint(self):
        outputPath = self.getNextCheckpointPath()
        update = self.nextUpdate - 1

        saveCheckpoint(
            self.model,
            self.optimizer,
            outputPath,
            update,
            self.numThinkSteps,
        )

        self.currentUpdate = update
        self.statusText = f"Saved policy update {update}"
        print(f"Saved training checkpoint: {outputPath}")

        return outputPath

    def setupBattle(self):
        self.engine = BattleEngine()
        self.battle = self.engine.create_battle()

        deck0, deck1 = sample_decks(self.decks, self.rng, mirror_match=False)

        self.currentDecks = {0: list(deck0), 1: list(deck1)}

        print(f"Player 0 deck: {deck0}")
        print(f"Player 1 deck: {deck1}")

        apply_deck_to_player(self.battle.players[0], deck0, self.rng)
        apply_deck_to_player(self.battle.players[1], deck1, self.rng)

        self.lastDecisionTick = self.battle.tick

        self.hiddenState = self.model.init_hidden_state(1, self.device)
        self.opponentHiddenState = self.opponentModel.init_hidden_state(1, self.device)

        self.pendingAction = None
        self.pendingLogits = None
        self.pendingMask = None
        self.scheduledAction = None
        self.scheduledActionAt = None

        self.noopUntilActive = False
        self.noopConditions = []

        self.pausedForDecision = False
        self.correctionMode = False
        self.conditionMenu = False
        self.correctionSlot = None
        self.gameOverMenu = False
        self.quitMenu = False

        self.lastTroopIds = self.getTroopIds()
        self.statusText = "Battle started"
        self.modelActionText = ""
        self.modelMarkerAction = None

    def getTroopIds(self):
        result = set()
        entities = getattr(self.battle, "entities", None)

        if entities is None:
            return result

        values = entities.values() if hasattr(entities, "values") else entities

        for entity in values:
            if not getattr(entity, "is_alive", True):
                continue

            entityType = str(getattr(entity, "entity_type", "")).lower()

            if entityType in {"tower", "projectile", "spell", "effect", "building", "area_effect"}:
                continue

            entityId = getattr(entity, "entity_id", None)
            if entityId is None:
                entityId = id(entity)

            result.add(entityId)

        return result

    def hasNewTroop(self):
        currentIds = self.getTroopIds()
        newIds = currentIds - self.lastTroopIds
        self.lastTroopIds = currentIds
        return bool(newIds)

    def getElixir(self):
        return float(getattr(self.battle.players[0], "elixir", 0.0))

    def conditionMet(self):
        if not self.noopConditions:
            return False

        for condition in self.noopConditions:
            if condition == "troop":
                if self.hasNewTroop():
                    return True
            elif condition.startswith("elixir:"):
                threshold = float(condition.split(":", 1)[1])
                if self.getElixir() >= threshold:
                    return True

        return False

    def buildObservation(self, playerId):
        observation = self.observationBuilder.build(self.battle, playerId)

        globalFeatures = torch.as_tensor(
            observation.global_features,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        board = torch.as_tensor(
            observation.board,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        entities = torch.as_tensor(
            observation.entities,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        entityMask = torch.as_tensor(
            observation.entity_mask,
            dtype=torch.bool,
            device=self.device,
        ).unsqueeze(0)

        return globalFeatures, board, entities, entityMask

    def getPolicyDecision(self, playerId, model, hiddenState, deterministic):
        mask = self.actionSpace.legal_action_mask(self.battle, playerId)
        legalActions = np.flatnonzero(mask)

        if len(legalActions) == 0:
            return NO_OP, None, mask, hiddenState

        globalFeatures, board, entities, entityMask = self.buildObservation(playerId)

        if playerId == 0:
            logits, _value, newHiddenState = model(
                globalFeatures,
                board,
                entities,
                entityMask,
                hiddenState,
            )
        else:
            with torch.no_grad():
                logits, _value, newHiddenState = model(
                    globalFeatures,
                    board,
                    entities,
                    entityMask,
                    hiddenState,
                )

        actionMask = torch.as_tensor(
            mask,
            dtype=torch.bool,
            device=self.device,
        ).unsqueeze(0)

        maskedLogits = logits.masked_fill(~actionMask, -1e9)

        if deterministic:
            action = int(maskedLogits.argmax(dim=-1).item())
        else:
            distribution = torch.distributions.Categorical(logits=maskedLogits)
            action = int(distribution.sample().item())

        if not mask[action]:
            action = NO_OP

        return action, logits, mask, newHiddenState

    def getModelDecision(self):
        action, logits, mask, newHiddenState = self.getPolicyDecision(
            playerId=0,
            model=self.model,
            hiddenState=self.hiddenState,
            deterministic=self.deterministic,
        )

        self.hiddenState = newHiddenState
        return action, logits, mask

    def getOpponentDecision(self):
        action, _logits, _mask, newHiddenState = self.getPolicyDecision(
            playerId=1,
            model=self.opponentModel,
            hiddenState=self.opponentHiddenState,
            deterministic=self.opponentDeterministic,
        )

        self.opponentHiddenState = newHiddenState.detach()
        return action

    def trainDecision(self, logits, targetAction):
        target = torch.tensor([targetAction], dtype=torch.long, device=self.device)
        loss = F.cross_entropy(logits, target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        self.hiddenState = self.hiddenState.detach()
        self.decisionCount += 1
        self.lastLoss = float(loss.item())

        if self.saveInterval > 0 and self.decisionCount % self.saveInterval == 0:
            self.saveTrainingCheckpoint()

        return self.lastLoss

    def applyAction(self, action):
        if action is None:
            return

        mask = self.actionSpace.legal_action_mask(self.battle, 0)

        if action < 0 or action >= len(mask) or not mask[action]:
            action = NO_OP

        self.actionSpace.apply_action(self.battle, 0, action)

    def finishDecision(self, targetAction):
        if self.pendingLogits is None:
            return

        if (
            targetAction != NO_OP
            and (
                self.pendingMask is None
                or targetAction >= len(self.pendingMask)
                or not self.pendingMask[targetAction]
            )
        ):
            return

        loss = self.trainDecision(self.pendingLogits, targetAction)

        self.scheduledAction = targetAction
        self.scheduledActionAt = time.monotonic() + self.decisionDelayMs / 1000.0

        if targetAction == NO_OP:
            self.statusText = f"NO-OP queued • {self.decisionDelayMs} ms ping"
        else:
            self.statusText = f"{self.actionName(targetAction)} • {self.decisionDelayMs} ms ping"

        self.lastLoss = loss
        self.pendingAction = None
        self.pendingLogits = None
        self.pendingMask = None

        self.pausedForDecision = False
        self.correctionMode = False
        self.correctionSlot = None

    def processScheduledAction(self):
        if self.scheduledAction is None or self.scheduledActionAt is None:
            return

        if time.monotonic() < self.scheduledActionAt:
            return

        action = self.scheduledAction
        self.scheduledAction = None
        self.scheduledActionAt = None

        self.applyAction(action)

        if action == NO_OP:
            self.statusText = "Played: NO-OP"
        else:
            self.statusText = f"Played: {self.actionName(action)}"

        self.modelMarkerAction = None

    def beginDecision(self):
        if self.pendingLogits is not None or self.scheduledAction is not None:
            return

        if self.noopUntilActive:
            if self.conditionMet():
                self.noopUntilActive = False
                self.noopConditions = []
                self.statusText = "Condition met — showing next model decision"
            else:
                action, logits, mask = self.getModelDecision()
                self.trainDecision(logits, NO_OP)
                self.applyAction(NO_OP)
                self.lastDecisionTick = self.battle.tick
                self.statusText = "AUTO NO-OP"
                return

        action, logits, mask = self.getModelDecision()

        self.pendingAction = action
        self.pendingLogits = logits
        self.pendingMask = mask
        self.modelMarkerAction = action
        if self.autoNoOp:
            self.finishDecision(NO_OP)
            return

        self.pausedForDecision = True
        self.statusText = "Choose what the model should have done"

    def actionName(self, action):
        if action == NO_OP:
            return "NO-OP"

        slot = action // TILE_COUNT
        tileIndex = action % TILE_COUNT
        tileY = tileIndex // BOARD_WIDTH
        tileX = tileIndex % BOARD_WIDTH
        hand = self.battle.players[0].hand
        card = hand[slot] if 0 <= slot < len(hand) else "Unknown"

        return f"{card}  •  Slot {slot + 1}  •  X={tileX}  Y={tileY}"

    def actionDetails(self, action):
        if action is None:
            return None

        if action == NO_OP:
            return {"card": "NO-OP", "slot": None, "x": None, "y": None}

        slot = action // TILE_COUNT
        tileIndex = action % TILE_COUNT
        tileY = tileIndex // BOARD_WIDTH
        tileX = tileIndex % BOARD_WIDTH
        hand = self.battle.players[0].hand
        card = hand[slot] if 0 <= slot < len(hand) else "Unknown"

        return {"card": card, "slot": slot + 1, "x": tileX, "y": tileY}

    def getLegalTiles(self, slot):
        if self.pendingMask is None:
            return set()

        base = slot * TILE_COUNT
        result = set()

        for tileIndex in range(TILE_COUNT):
            action = base + tileIndex
            if action < len(self.pendingMask) and self.pendingMask[action]:
                result.add(tileIndex)

        return result

    def actionFromMouse(self, position):
        if self.correctionSlot is None:
            return None

        x, y = position
        arenaRect = pygame.Rect(ARENA_X, ARENA_Y, ARENA_WIDTH, ARENA_HEIGHT)

        if not arenaRect.collidepoint(x, y):
            return None

        screenTileX = int((x - ARENA_X) / TILE_SIZE)
        screenTileY = int((y - ARENA_Y) / TILE_SIZE)

        if not (0 <= screenTileX < BOARD_WIDTH and 0 <= screenTileY < BOARD_HEIGHT):
            return None

        # Action tiles are in player 0's view frame, which is exactly the
        # on-screen grid (enemy at the top, own side at the bottom).
        tileIndex = screenTileY * BOARD_WIDTH + screenTileX
        action = self.correctionSlot * TILE_COUNT + tileIndex

        if self.pendingMask is not None and self.pendingMask[action]:
            return action

        return None

    def startNoopUntil(self):
        if self.pendingLogits is None:
            return

        self.finishDecision(NO_OP)

        self.noopUntilActive = True
        self.noopConditions = []
        self.conditionMenu = True
        self.pausedForDecision = True
        self.statusText = "Select one or more conditions"

    def chooseCondition(self, condition):
        if condition in self.noopConditions:
            self.noopConditions.remove(condition)
        else:
            self.noopConditions.append(condition)

    def handleConditionKey(self, key):
        conditionMap = {
            pygame.K_1: "elixir:5",
            pygame.K_2: "elixir:6",
            pygame.K_3: "elixir:7",
            pygame.K_4: "elixir:8",
            pygame.K_5: "elixir:9",
            pygame.K_6: "elixir:10",
        }

        if key in conditionMap:
            self.chooseCondition(conditionMap[key])
            return

        if key == pygame.K_t:
            self.chooseCondition("troop")
            return

        if key == pygame.K_RETURN:
            if self.noopConditions:
                self.conditionMenu = False
                self.pausedForDecision = False
                self.statusText = "NO-OP until " + " OR ".join(
                    self.noopConditionName(condition)
                    for condition in self.noopConditions
                )
            return

        if key == pygame.K_ESCAPE:
            self.conditionMenu = False
            self.noopUntilActive = False
            self.noopConditions = []
            self.pausedForDecision = False
            self.statusText = "NO-OP until cancelled"

    def noopConditionName(self, condition):
        if condition == "troop":
            return "new troop"
        return "elixir >= " + condition.split(":", 1)[1]

    def handleQuitKey(self, key):
        if key == pygame.K_q:
            self.saveTrainingCheckpoint()
            self.running = False
            return

        if key == pygame.K_ESCAPE:
            self.quitMenu = False
            self.pausedForDecision = False
            self.statusText = "Quit cancelled"

    def handleEvents(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.saveTrainingCheckpoint()
                self.running = False
                return

            if event.type == pygame.KEYDOWN:
                if self.gameOverMenu:
                    if event.key == pygame.K_r:
                        self.setupBattle()
                        continue

                    if event.key == pygame.K_q:
                        self.saveTrainingCheckpoint()
                        self.running = False
                        continue

                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                        continue

                    continue

                if self.quitMenu:
                    self.handleQuitKey(event.key)
                    continue

                if event.key == pygame.K_ESCAPE:
                    if self.conditionMenu:
                        self.handleConditionKey(event.key)
                    else:
                        self.quitMenu = True
                        self.pausedForDecision = True
                        self.statusText = "Quit menu — Q saves and exits"
                    continue

                if self.conditionMenu:
                    self.handleConditionKey(event.key)
                    continue

                if event.key == pygame.K_SPACE:
                    self.autoNoOp = not self.autoNoOp

                    if self.autoNoOp and self.pendingLogits is not None:
                        self.finishDecision(NO_OP)

                    self.pausedForDecision = not self.autoNoOp
                    return

                if self.pendingLogits is None:
                    continue

                if event.key == pygame.K_a:
                    self.finishDecision(self.pendingAction)
                    continue

                if event.key == pygame.K_n:
                    self.finishDecision(NO_OP)
                    continue

                if event.key == pygame.K_c:
                    self.correctionMode = True
                    self.statusText = "Press 1-4, then click a legal tile"
                    continue

                if event.key == pygame.K_u:
                    self.startNoopUntil()
                    continue

                if self.correctionMode and pygame.K_1 <= event.key <= pygame.K_4:
                    slot = event.key - pygame.K_1

                    if slot >= len(self.battle.players[0].hand):
                        continue

                    if not self.getLegalTiles(slot):
                        self.statusText = "That card has no legal placement"
                        continue

                    self.correctionSlot = slot
                    self.statusText = f"Click a legal tile for {self.battle.players[0].hand[slot]}"

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.correctionMode:
                action = self.actionFromMouse(event.pos)
                if action is not None:
                    self.finishDecision(action)

    def drawModelMarker(self):
        if self.modelMarkerAction is None or self.modelMarkerAction == NO_OP:
            return

        details = self.actionDetails(self.modelMarkerAction)
        if details is None:
            return

        tileX = details["x"]
        tileY = details["y"]

        rect = pygame.Rect(
            ARENA_X + tileX * TILE_SIZE,
            ARENA_Y + tileY * TILE_SIZE,
            TILE_SIZE,
            TILE_SIZE,
        )
        centerX, centerY = rect.center

        pygame.draw.rect(self.screen, (255, 215, 0), rect, 3)
        pygame.draw.circle(self.screen, (255, 215, 0), (centerX, centerY), 9, 3)
        pygame.draw.line(self.screen, (255, 215, 0), (centerX - 12, centerY), (centerX + 12, centerY), 2)
        pygame.draw.line(self.screen, (255, 215, 0), (centerX, centerY - 12), (centerX, centerY + 12), 2)

    def drawCorrectionTiles(self):
        if not self.correctionMode or self.correctionSlot is None:
            return

        legalTiles = self.getLegalTiles(self.correctionSlot)

        for tileIndex in legalTiles:
            tileY = tileIndex // BOARD_WIDTH
            tileX = tileIndex % BOARD_WIDTH
            screenY = ARENA_Y + tileY * TILE_SIZE

            rect = pygame.Rect(
                ARENA_X + tileX * TILE_SIZE,
                screenY,
                TILE_SIZE,
                TILE_SIZE,
            )

            overlay = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
            overlay.fill((80, 255, 120, 80))
            self.screen.blit(overlay, rect)
            pygame.draw.rect(self.screen, (80, 255, 120), rect, 1)

    def drawPanel(self, rect, title, accent=UI_GOLD):
        pygame.draw.rect(self.screen, UI_BG, rect, border_radius=12)
        pygame.draw.rect(self.screen, UI_BORDER, rect, 2, border_radius=12)

        titleSurface = self.trainerFont.render(title, True, accent)
        self.screen.blit(titleSurface, (rect.x + 16, rect.y + 12))

    def drawKeyRow(self, x, y, key, text, active=False):
        keyRect = pygame.Rect(x, y, 42, 28)
        keyFill = UI_GOLD_DARK if active else UI_BG_LIGHT

        pygame.draw.rect(self.screen, keyFill, keyRect, border_radius=6)
        pygame.draw.rect(self.screen, UI_BORDER, keyRect, 1, border_radius=6)

        keySurface = self.trainerSmallFont.render(key, True, UI_TEXT)
        self.screen.blit(keySurface, keySurface.get_rect(center=keyRect.center))

        textSurface = self.trainerSmallFont.render(text, True, UI_TEXT_DIM)
        self.screen.blit(textSurface, (x + 54, y + 4))

    def drawConditionMenu(self, x, y):
        panelRect = pygame.Rect(x - 12, y - 12, PANEL_WIDTH - 24, 390)
        self.drawPanel(panelRect, "NO-OP UNTIL", UI_GOLD)
        y += 38

        lines = [
            ("1", "Elixir >= 5", "elixir:5"),
            ("2", "Elixir >= 6", "elixir:6"),
            ("3", "Elixir >= 7", "elixir:7"),
            ("4", "Elixir >= 8", "elixir:8"),
            ("5", "Elixir >= 9", "elixir:9"),
            ("6", "Elixir = 10", "elixir:10"),
            ("T", "New troop appears", "troop"),
        ]

        for key, description, condition in lines:
            self.drawKeyRow(x, y, key, description, condition in self.noopConditions)
            y += 38

        pygame.draw.line(
            self.screen,
            UI_BORDER,
            (x, y + 2),
            (panelRect.right - 12, y + 2),
            1,
        )

        y += 16
        selected = " OR ".join(self.noopConditionName(c) for c in self.noopConditions)

        selectedSurface = self.trainerSmallFont.render(
            f"Selected: {selected or 'None'}",
            True,
            UI_TEXT,
        )
        self.screen.blit(selectedSurface, (x, y))

        y += 34
        self.drawKeyRow(x, y, "↵", "Confirm", bool(self.noopConditions))
        y += 38
        self.drawKeyRow(x, y, "ESC", "Cancel")

    def drawOverlayBox(self, title, rows, accent=UI_GOLD, height=250):
        overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 175))
        self.screen.blit(overlay, (0, 0))

        boxWidth = 500
        box = pygame.Rect(
            (self.screen.get_width() - boxWidth) // 2,
            (self.screen.get_height() - height) // 2,
            boxWidth,
            height,
        )

        self.drawPanel(box, title, accent)

        y = box.y + 62

        for key, text in rows:
            self.drawKeyRow(box.x + 28, y, key, text)
            y += 38

    def drawGameOverMenu(self):
        winner = getattr(self.battle, "winner", None)

        rows = [
            ("R", "Restart game"),
            ("Q", "Save & quit"),
            ("ESC", "Quit without saving"),
        ]

        self.drawOverlayBox("GAME OVER", rows, UI_GOLD, 250)

        if winner is not None:
            text = self.trainerSmallFont.render(
                f"Winner: Player {winner}",
                True,
                UI_TEXT,
            )
            textRect = text.get_rect(
                center=(self.screen.get_width() // 2, 285)
            )
            self.screen.blit(text, textRect)

    def drawQuitMenu(self):
        self.drawOverlayBox(
            "QUIT TRAINING",
            [
                ("Q", "Save checkpoint & quit"),
                ("ESC", "Cancel"),
            ],
            UI_GOLD,
            210,
        )

    def drawTrainerUI(self):
        panelRect = pygame.Rect(PANEL_X, PANEL_Y, PANEL_WIDTH, PANEL_HEIGHT)
        self.drawPanel(panelRect, "HAND TRAINER", UI_GOLD)

        x = panelRect.x + 16
        y = panelRect.y + 54

        updateSurface = self.trainerSmallFont.render(
            f"Policy update  {self.currentUpdate}  →  {self.nextUpdate}",
            True,
            UI_TEXT,
        )
        self.screen.blit(updateSurface, (x, y))
        y += 28

        thinkSurface = self.trainerSmallFont.render(
            f"Thinking steps: {self.numThinkSteps}",
            True,
            UI_TEXT_DIM,
        )
        self.screen.blit(thinkSurface, (x, y))
        y += 30

        if self.pendingAction is not None:
            details = self.actionDetails(self.pendingAction)
            card = details["card"]

            self.drawPanel(
                pygame.Rect(x - 4, y - 4, panelRect.width - 24, 155),
                "MODEL DECISION",
                BLUE,
            )

            y += 38

            actionText = self.trainerFont.render(card, True, UI_TEXT)
            self.screen.blit(actionText, (x + 4, y))
            y += 31

            if card != "NO-OP":
                detail = self.trainerSmallFont.render(
                    f"Slot {details['slot']}   X {details['x']}   Y {details['y']}",
                    True,
                    UI_TEXT_DIM,
                )
            else:
                detail = self.trainerSmallFont.render(
                    "No card placement",
                    True,
                    UI_TEXT_DIM,
                )

            self.screen.blit(detail, (x + 4, y))
            y += 43

            for key, text in [
                ("A", "Accept"),
                ("SPACE", "NO-OP"),
                ("C", "Change"),
                ("U", "NO-OP until"),
            ]:
                self.drawKeyRow(x, y, key, text)
                y += 31

        elif self.scheduledAction is not None:
            details = self.actionDetails(self.scheduledAction)
            card = details["card"]

            self.drawPanel(
                pygame.Rect(x - 4, y - 4, panelRect.width - 24, 125),
                "PING",
                UI_GOLD,
            )

            y += 38

            actionText = self.trainerFont.render(card, True, UI_TEXT)
            self.screen.blit(actionText, (x + 4, y))
            y += 35

            remainingMs = max(
                0,
                int((self.scheduledActionAt - time.monotonic()) * 1000),
            )

            pingText = self.trainerSmallFont.render(
                f"Playing in {remainingMs} ms",
                True,
                UI_TEXT_DIM,
            )
            self.screen.blit(pingText, (x + 4, y))

        elif self.noopUntilActive:
            self.drawPanel(
                pygame.Rect(x - 4, y - 4, panelRect.width - 24, 150),
                "AUTO NO-OP",
                UI_GOLD,
            )

            y += 38

            text = self.trainerSmallFont.render(
                "Waiting for:",
                True,
                UI_TEXT,
            )
            self.screen.blit(text, (x + 4, y))
            y += 28

            for condition in self.noopConditions:
                text = self.trainerSmallFont.render(
                    "• " + self.noopConditionName(condition),
                    True,
                    UI_TEXT,
                )
                self.screen.blit(text, (x + 4, y))
                y += 24

        elif self.conditionMenu:
            self.drawConditionMenu(x, y)

        elif self.correctionMode:
            self.drawPanel(
                pygame.Rect(x - 4, y - 4, panelRect.width - 24, 135),
                "CHANGE ACTION",
                BLUE,
            )

            y += 40
            self.drawKeyRow(x, y, "1-4", "Choose card slot")
            y += 36
            self.drawKeyRow(x, y, "MOUSE", "Click legal tile")

        else:
            self.drawPanel(
                pygame.Rect(x - 4, y - 4, panelRect.width - 24, 170),
                "CONTROLS",
                UI_GOLD,
            )

            y += 40

            for key, text in [
                ("SPACE", "NO-OP"),
                ("U", "NO-OP until condition"),
                ("ESC", "Quit menu"),
            ]:
                self.drawKeyRow(x, y, key, text)
                y += 35

        statsY = panelRect.bottom - 178

        pygame.draw.line(
            self.screen,
            UI_BORDER,
            (x, statsY - 10),
            (panelRect.right - 16, statsY - 10),
            1,
        )

        stats = [
            f"Elixir   {self.getElixir():.1f} / 10",
            f"Decisions   {self.decisionCount}",
            f"Battle   {getattr(self.battle, 'time', 0.0):.1f}s",
            f"Loss   {self.lastLoss:.4f}" if self.lastLoss is not None else "Loss   —",
            f"Ping   {self.decisionDelayMs} ms",
        ]

        for line in stats:
            text = self.trainerSmallFont.render(line, True, UI_TEXT)
            self.screen.blit(text, (x, statsY))
            statsY += 25

        status = self.statusText or "Ready"
        statusSurface = self.trainerSmallFont.render(status[:48], True, UI_TEXT_DIM)
        self.screen.blit(statusSurface, (x, panelRect.bottom - 35))

    def draw(self):
        self.draw_backdrop()
        self.draw_arena()
        self.drawCorrectionTiles()
        self.draw_towers()
        self.draw_entities()
        self.drawModelMarker()
        self.draw_hands()
        self.drawTrainerUI()

        if self.gameOverMenu:
            self.drawGameOverMenu()
        elif self.quitMenu:
            self.drawQuitMenu()

        pygame.display.flip()

    def stepBattle(self):
        if self.battle.game_over:
            self.gameOverMenu = True
            self.pausedForDecision = True
            return

        self.battle.step(speed_factor=1.0)

        if self.battle.game_over:
            self.gameOverMenu = True
            self.pausedForDecision = True
            self.statusText = "Battle finished"

    def maybeTakeActions(self):
        if self.conditionMenu or self.gameOverMenu or self.quitMenu:
            return

        if self.pendingLogits is not None or self.scheduledAction is not None:
            return

        if self.battle.game_over:
            self.gameOverMenu = True
            self.pausedForDecision = True
            return

        if self.battle.tick - self.lastDecisionTick < self.decisionInterval:
            return

        opponentAction = self.getOpponentDecision()
        opponentMask = self.actionSpace.legal_action_mask(self.battle, 1)

        if (
            opponentAction < 0
            or opponentAction >= len(opponentMask)
            or not opponentMask[opponentAction]
        ):
            opponentAction = NO_OP

        self.actionSpace.apply_action(self.battle, 1, opponentAction)
        self.beginDecision()
        self.lastDecisionTick = self.battle.tick

    def run(self):
        self.running = True

        while self.running:
            self.handleEvents()

            if not self.gameOverMenu and not self.quitMenu:
                self.maybeTakeActions()

                if self.pendingLogits is None:
                    self.processScheduledAction()

                if self.pendingLogits is None:
                    self.stepBattle()

            self.draw()
            self.clock.tick(60)

        pygame.quit()


def parseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Initial training checkpoint. With --continue, latest is used if omitted.",
    )
    parser.add_argument(
        "--opponent-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint used by the opponent. Defaults to the first loaded checkpoint.",
    )
    parser.add_argument(
        "--continue",
        dest="continueTraining",
        action="store_true",
        help="Continue from the latest checkpoint in --output-dir.",
    )
    parser.add_argument("--decks-path", type=Path, default=DECKS_FILE)
    parser.add_argument("--decision-interval", type=int, default=8)
    parser.add_argument(
        "--decision-delay-ms",
        type=int,
        default=200,
        help="Delay between the human decision and actually playing the action.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use argmax for the hand-trained model.",
    )
    parser.add_argument(
        "--opponent-deterministic",
        action="store_true",
        help="Use argmax for the opponent model.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed. If omitted, use a random seed each run.",
    )
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--save-interval", type=int, default=25)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional exact initial/output path. Incremental saves still use --output-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for incremental policy_update_XXXX.pt checkpoints.",
    )
    parser.add_argument(
        "--num-think-steps",
        type=int,
        default=0,
        help="Number of ThinkingBlock refinement passes used by both models.",
    )

    return parser.parse_args()


def main():
    args = parseArgs()

    if args.num_think_steps < 0:
        raise ValueError("--num-think-steps must be >= 0")

    if args.decision_delay_ms < 0:
        raise ValueError("--decision-delay-ms must be >= 0")

    if args.output is not None and args.output_dir is not None:
        raise ValueError("Use either --output or --output-dir, not both.")

    if args.output_dir is not None:
        outputDir = args.output_dir.resolve()
    elif args.output is not None:
        outputDir = args.output.resolve().parent
    elif args.checkpoint is not None:
        outputDir = args.checkpoint.resolve().parent
    else:
        raise ValueError("--output-dir is required when --checkpoint is omitted")

    latestCheckpoint, latestUpdate = getLatestCheckpoint(outputDir)

    explicitCheckpoint = args.checkpoint.resolve() if args.checkpoint is not None else None
    explicitOpponent = (
        args.opponent_checkpoint.resolve()
        if args.opponent_checkpoint is not None
        else None
    )

    if args.continueTraining:
        if latestCheckpoint is not None:
            trainingCheckpointPath = latestCheckpoint
            print(f"Resuming latest hand-training checkpoint: {latestCheckpoint}")
        elif explicitCheckpoint is not None:
            trainingCheckpointPath = explicitCheckpoint
            print(f"No checkpoint found in {outputDir}; starting from {explicitCheckpoint}")
        else:
            raise FileNotFoundError(f"No policy_update_*.pt checkpoint found in {outputDir}")
    else:
        if explicitCheckpoint is None:
            raise ValueError("--checkpoint is required unless --continue is used")
        trainingCheckpointPath = explicitCheckpoint

    if not trainingCheckpointPath.exists():
        raise FileNotFoundError(f"Training checkpoint does not exist: {trainingCheckpointPath}")

    if explicitOpponent is not None:
        opponentCheckpointPath = explicitOpponent
    elif explicitCheckpoint is not None:
        opponentCheckpointPath = explicitCheckpoint
    else:
        opponentCheckpointPath = trainingCheckpointPath

    if not opponentCheckpointPath.exists():
        raise FileNotFoundError(f"Opponent checkpoint does not exist: {opponentCheckpointPath}")

    if args.output is not None:
        outputPath = args.output.resolve()

        if outputPath != trainingCheckpointPath and not outputPath.exists():
            outputPath.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(trainingCheckpointPath, outputPath)

    device = resolve_torch_device(args.device)

    print(f"Training checkpoint: {trainingCheckpointPath}")
    print(f"Opponent checkpoint: {opponentCheckpointPath}")
    print(f"Checkpoint directory: {outputDir}")
    print(f"Device: {device}")
    print(f"Continue training: {args.continueTraining}")
    print(f"Num think steps: {args.num_think_steps}")
    print(f"Decision delay: {args.decision_delay_ms} ms")
    print(f"Next checkpoint update: {getCheckpointUpdate(trainingCheckpointPath) + 1}")

    trainer = HandTrainer(
        checkpointPath=trainingCheckpointPath,
        outputDir=outputDir,
        opponentCheckpointPath=opponentCheckpointPath,
        decksPath=args.decks_path,
        decisionInterval=args.decision_interval,
        decisionDelayMs=args.decision_delay_ms,
        device=device,
        deterministic=args.deterministic,
        opponentDeterministic=args.opponent_deterministic,
        seed=args.seed,
        learningRate=args.lr,
        saveInterval=args.save_interval,
        continueTraining=True,
        numThinkSteps=args.num_think_steps,
    )

    trainer.run()


if __name__ == "__main__":
    main()
