"""Czysty silnik reguł Paper Soccer (bez zależności od bibliotek RL).

To jest refaktor ``legacy/env_original.py``. Zachowanie reguł jest IDENTYCZNE
(pilnuje tego ``tests/test_golden.py``), doszły tylko:

* ``legal_actions()`` / ``action_masks()``,
* ``get_state()`` / ``set_state()`` / ``copy()`` / ``peek()`` (potrzebne Dyna i zapisowi partii),
* ``state_key()`` - zwarty, hashowalny klucz stanu dla agenta tablicowego,
* powód zakończenia gry (``termination_reason``) i zwycięzca (``winner``),
* ``render()`` - podgląd ASCII do debugowania.

Konwencje (jak w oryginale):

* ``grid[y, x, 0:8]`` - 1 = akcja z tego punktu NIEDOSTĘPNA (krawędź zużyta / ściana),
* ``grid[y, x, 8]``   - 1 = punkt brzegowy lub już odwiedzony (ruch na niego = odbicie),
* gracze to ``-1`` (zaczyna) i ``+1``; gracz ``+1`` strzela gola w górną bramkę (wiersz 0),
  gracz ``-1`` w dolną (ostatni wiersz).
"""
from __future__ import annotations

import numpy as np

# kierunki: (dy, dx)
ACTION_MAP = {
    0: (-1, -1),
    1: (-1, 0),
    2: (-1, 1),
    3: (0, 1),
    4: (1, 1),
    5: (1, 0),
    6: (1, -1),
    7: (0, -1),
}
N_ACTIONS = 8
PLAYERS = (-1, 1)


def parse_board(spec: str) -> tuple[int, int]:
    """'8x10' -> (x_length=8, y_length=10)."""
    x, y = spec.lower().split("x")
    return int(x), int(y)


class PaperSoccerGame:
    def __init__(self, x_length: int = 8, y_length: int = 10, gate_size: int = 2):
        if not (x_length % 2 == 0 and y_length % 2 == 0 and gate_size % 2 == 0):
            raise ValueError("x_length, y_length i gate_size muszą być parzyste")
        if not gate_size < y_length:
            raise ValueError("gate_size musi być mniejszy niż y_length")

        self.x_length = x_length
        self.y_length = y_length
        self.gate_size = gate_size
        self.ACTION_MAP = ACTION_MAP
        self.start_pos = ((y_length + 3) // 2, (x_length + 1) // 2)

        # szablon początkowej planszy (liczony raz; reset() tylko go kopiuje)
        self._template = self._build_template()

        self.grid = self._template.copy()
        self.cur_pos = list(self.start_pos)
        self.current_player = -1
        self.done = False
        self.winner = 0
        self.termination_reason: str | None = None

    # ------------------------------------------------------------------ budowa planszy
    def _build_template(self) -> np.ndarray:
        grid = np.zeros((self.y_length + 3, self.x_length + 1, 9), dtype=np.int8)
        self._create_boundaries(grid)
        self._set_possible_actions(grid)
        grid[self.start_pos[0], self.start_pos[1], 8] = 1  # start jako punkt odbicia
        return grid

    def _create_boundaries(self, grid):
        left = self.x_length // 2 - self.gate_size // 2
        right = self.x_length // 2 + self.gate_size // 2
        grid[0, :, 8] = 1
        grid[-1, :, 8] = 1

        grid[:, 0, 8] = 1
        grid[:, -1, 8] = 1

        grid[[1, -2], :left + 1, 8] = 1
        grid[[1, -2], right:, 8] = 1

    def _set_possible_actions(self, grid):
        left = self.x_length // 2 - self.gate_size // 2
        right = self.x_length // 2 + self.gate_size // 2

        # left band without corners
        grid[2:-2, 0, [0, 1, 5, 6, 7, 8]] = 1
        # right band with corners
        grid[2:-2, -1, [1, 2, 3, 4, 5, 8]] = 1

        # left upper corner
        grid[1, 0, [0, 1, 2, 3, 5, 6, 7, 8]] = 1
        # right upper corner
        grid[1, -1, [0, 1, 2, 3, 4, 5, 7, 8]] = 1
        # left down corner
        grid[-2, 0, [0, 1, 3, 4, 5, 6, 7, 8]] = 1
        # right down corner
        grid[-2, -1, [1, 2, 3, 4, 5, 6, 7, 8]] = 1

        grid[1, 1:left, [0, 1, 2, 3, 7, 8]] = 1
        grid[1, right + 1:-1, [0, 1, 2, 3, 7, 8]] = 1
        grid[-2, 1:left, [3, 4, 5, 6, 7, 8]] = 1
        grid[-2, right + 1:-1, [3, 4, 5, 6, 7, 8]] = 1

        grid[1, left, [0, 1, 7, 8]] = 1
        grid[1, right, [1, 2, 3, 8]] = 1
        grid[-2, left, [5, 6, 7, 8]] = 1
        grid[-2, right, [3, 4, 5, 8]] = 1

        grid[0, left + 1:right, [0, 1, 2, 3, 7, 8]] = 1
        grid[-1, left + 1:right, [3, 4, 5, 6, 7, 8]] = 1

        # unavailable areas
        grid[0, :left, :] = 1
        grid[0, right + 1:, :] = 1
        grid[-1, :left, :] = 1
        grid[-1, right + 1:, :] = 1

        # rear corners
        grid[0, left, [0, 1, 2, 3, 5, 6, 7, 8]] = 1
        grid[0, right, [0, 1, 2, 3, 4, 5, 7, 8]] = 1

        grid[-1, left, [0, 1, 3, 4, 5, 6, 7, 8]] = 1
        grid[-1, right, [1, 2, 3, 4, 5, 6, 7, 8]] = 1

    # ------------------------------------------------------------------ API
    def reset(self):
        self.grid = self._template.copy()
        self.cur_pos = list(self.start_pos)
        self.current_player = -1
        self.done = False
        self.winner = 0
        self.termination_reason = None
        return self.grid.copy(), self.current_player

    def goal_row(self, player: int) -> int:
        """Wiersz, do którego ``player`` musi doprowadzić piłkę, żeby wygrać."""
        return 0 if player == 1 else self.grid.shape[0] - 1

    def action_masks(self) -> np.ndarray:
        """bool[8], True = akcja legalna."""
        y, x = self.cur_pos
        return self.grid[y, x, :N_ACTIONS] == 0

    def legal_actions(self) -> np.ndarray:
        y, x = self.cur_pos
        return np.flatnonzero(self.grid[y, x, :N_ACTIONS] == 0)

    def step(self, action: int, copy_grid: bool = True):
        """Wykonuje ruch. Zwraca (grid, rewards, end, current_player) jak oryginał.

        ``copy_grid=False`` pomija kopiowanie planszy (szybciej; zwracany grid to wtedy widok).
        """
        end = False
        rewards = {1: 0, -1: 0}
        y, x = self.cur_pos
        out = (lambda: self.grid.copy()) if copy_grid else (lambda: self.grid)

        if self.grid[y, x, action] == 1:  # akcja niedostępna -> porażka gracza
            self._finish(rewards, loser=self.current_player, reason="illegal_move")
            return out(), rewards, True, self.current_player

        dy, dx = ACTION_MAP[action]
        self.grid[y, x, action] = 1
        new_y, new_x = y + dy, x + dx
        self.cur_pos = [new_y, new_x]
        opposite_action = (action - 4) % 8
        self.grid[new_y, new_x, opposite_action] = 1

        if new_y == 0:
            end = True
            self._finish(rewards, winner=1, reason="goal_top")
        elif new_y == self.grid.shape[0] - 1:
            end = True
            self._finish(rewards, winner=-1, reason="goal_bottom")
        elif np.all(self.grid[new_y, new_x, 0:8] == 1):
            end = True
            self._finish(rewards, loser=self.current_player, reason="blocked")
        else:
            if self.grid[new_y, new_x, 8] == 0:
                self.current_player *= -1
                self.grid[new_y, new_x, 8] = 1

        return out(), rewards, end, self.current_player

    def _finish(self, rewards, winner=None, loser=None, reason=None):
        if winner is None:
            winner = -loser
        rewards[winner] = 1
        rewards[-winner] = -1
        self.done = True
        self.winner = winner
        self.termination_reason = reason

    # ------------------------------------------------------------------ stan / kopie
    def get_state(self):
        """Pełna informacja potrzebna do odtworzenia pozycji (bez historii ruchów)."""
        return (self.grid.copy(), tuple(self.cur_pos), self.current_player)

    def set_state(self, state):
        grid, pos, player = state
        self.grid[...] = grid
        self.cur_pos = list(pos)
        self.current_player = player
        self.done = False
        self.winner = 0
        self.termination_reason = None

    def copy(self) -> "PaperSoccerGame":
        g = object.__new__(PaperSoccerGame)
        g.__dict__.update(self.__dict__)
        g.grid = self.grid.copy()
        g.cur_pos = list(self.cur_pos)
        return g  # _template jest współdzielony (tylko do odczytu)

    def peek(self, action: int):
        """Symulacja ruchu na kopii: (nowa_gra, rewards, end). Oryginał nietknięty."""
        g = self.copy()
        _, rewards, end, _ = g.step(action, copy_grid=False)
        return g, rewards, end

    def state_key(self, include_player: bool = True) -> bytes:
        """Zwarty klucz stanu (bity planszy + pozycja [+ gracz]) do słownika Q.

        Gracz jest domyślnie w kluczu: ta sama plansza ma inną wartość dla
        gracza strzelającego w górę i w dół.
        """
        key = np.packbits(self.grid).tobytes() + bytes(self.cur_pos)
        if include_player:
            key += b"\x01" if self.current_player == 1 else b"\x00"
        return key

    # ------------------------------------------------------------------ debug
    def grid_overview(self):
        print(self.grid[:, :, 8])

    def render(self) -> str:
        """Podgląd ASCII: O piłka, + odwiedzony punkt, # punkt brzegowy, . wolny punkt."""
        H, W = self.grid.shape[:2]
        t = self._template
        canvas = [[" "] * (2 * W - 1) for _ in range(2 * H - 1)]
        for y in range(H):
            for x in range(W):
                if t[y, x, :N_ACTIONS].all() and t[y, x, 8]:
                    continue  # poza boiskiem
                if [y, x] == self.cur_pos:
                    ch = "O"
                elif t[y, x, 8]:
                    ch = "#"
                elif self.grid[y, x, 8]:
                    ch = "+"
                else:
                    ch = "."
                canvas[2 * y][2 * x] = ch
        glyph = {0: "\\", 1: "|", 2: "/", 3: "-", 4: "\\", 5: "|", 6: "/", 7: "-"}
        played = (self.grid[:, :, :N_ACTIONS] == 1) & (t[:, :, :N_ACTIONS] == 0)
        for y, x, a in zip(*np.nonzero(played)):
            dy, dx = ACTION_MAP[int(a)]
            canvas[2 * y + dy][2 * x + dx] = glyph[int(a)]
        lines = ["".join(r).rstrip() for r in canvas]
        who = "koniec: " + str(self.termination_reason) if self.done else f"na ruchu: {self.current_player}"
        return "\n".join(lines) + f"\n[{who}]"


# alias zgodny z nazwą z planu / oryginału
PaperSoccerEnv = PaperSoccerGame
