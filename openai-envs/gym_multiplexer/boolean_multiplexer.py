import random

from gym.spaces import Discrete

from .multiplexer import Multiplexer
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class BooleanMultiplexer(Multiplexer):

    def __init__(self, control_bits=3) -> None:
        super().__init__(control_bits)
        self.observation_space = Discrete(self._observation_string_length)
        self.action_space = Discrete(2)

    def _generate_state(self):
        return [random.randint(0, 1) for _ in
                range(0, self._observation_string_length - 1)]

    def _internal_state(self):
        return list(map(lambda x: round(x), self._state))

    def save_grid(self, filename):
        self.reset()
        if not self._state:
            print(
                "Błąd: Stan jest pusty. Wywołaj env.reset() przed renderowaniem.")
            return

        # Wysokość okna rośnie dynamicznie wraz z liczbą bitów
        fig, ax = plt.subplots(figsize=(11, max(6, self.control_bits * 2.5)))

        num_addr = self.control_bits
        num_data = 2 ** num_addr

        addr_bits = self._state[:num_addr]
        data_bits = self._state[num_addr: num_addr + num_data]
        active_idx = int("".join(map(str, addr_bits)), 2)

        # --- 1. IDEALNE PROPORCJE TRAPEZU (MUX) ---
        x_left = 3.5
        x_right = 7.5

        # Dopasowujemy wysokość obudowy, żeby kable wchodziły z marginesami
        y_top_left = num_data
        y_bottom_left = -1

        # Prawa strona ukosuje się łagodnie (ścina po 20% z góry i dołu)
        height_left = y_top_left - y_bottom_left
        y_top_right = y_top_left - (height_left * 0.2)
        y_bottom_right = y_bottom_left + (height_left * 0.2)

        mux_poly = patches.Polygon([
            (x_left, y_bottom_left), (x_left, y_top_left),
            (x_right, y_top_right), (x_right, y_bottom_right)
        ], closed=True, facecolor='darkblue', edgecolor='#111111',
            linewidth=2.5, zorder=2)
        ax.add_patch(mux_poly)

        # Elegancki podpis na środku
        ax.text((x_left + x_right) / 2, (y_top_left + y_bottom_left) / 2,
                f"MUX\n{num_data}:1", color="white", fontsize=15,
                ha="center", va="center", fontweight="bold", zorder=3)

        # --- 2. WEJŚCIA DANYCH (Z lewej strony) ---
        for i in range(num_data):
            y = (num_data - 1) - i  # Rysujemy od góry do dołu
            is_active = (i == active_idx)

            # Ładniejszy, chłodny szary dla nieaktywnych
            color = "gold" if is_active else "#b2bec3"
            lw = 5.0 if is_active else 2.0

            # Linia sygnału
            ax.plot([0, x_left], [y, y], color=color, linewidth=lw, zorder=1)

            # Opis wartości na zewnątrz
            font_color = "darkgoldenrod" if is_active else "#2d3436"
            ax.text(1.5, y + 0.25, f"D{i} = {data_bits[i]}",
                    fontsize=12, fontweight="bold", color=font_color,
                    ha="center")

            # Malutki numerek pinu WEWNĄTRZ trapezu (inżynierski sznyt)
            ax.text(x_left + 0.2, y, str(i), color="white", fontsize=9,
                    va="center", fontweight="bold")

        # --- 3. BITY ADRESOWE (Od dołu) ---
        for i in range(num_addr):
            # Rozkładamy je równomiernie pod układem
            x = x_left + 1.0 + i * (
                    (x_right - x_left - 2.0) / max(1, (num_addr - 1)))

            # Idealne wyliczenie, gdzie linia ma dotknąć skośnej krawędzi
            y_edge = y_bottom_left + (x - x_left) * (
                    (y_bottom_right - y_bottom_left) / (x_right - x_left))

            ax.plot([x, x], [-3, y_edge], color="black", linewidth=2.5,
                    zorder=1)

            # Opis pinu na zewnątrz
            ax.text(x, -3.6, f"A{i} = {addr_bits[i]}",
                    fontsize=12, fontweight="bold", color="black", ha="center")

            # Malutki podpis bitu sterującego WEWNĄTRZ układu
            ax.text(x, y_edge + 0.2, str(i), color="white", fontsize=9,
                    ha="center", fontweight="bold")

        # --- 4. WYJŚCIE UKŁADU (Z prawej strony) ---
        out_val = data_bits[active_idx]
        y_out = (y_top_right + y_bottom_right) / 2

        ax.plot([x_right, 11], [y_out, y_out], color="gold", linewidth=6,
                zorder=1)
        ax.text(9.5, y_out + 0.35, f"OUT = {out_val}",
                fontsize=16, fontweight="bold", color="darkgoldenrod",
                ha="center")

        # --- USTAWIENIA OKNA ---
        ax.set_xlim(-0.5, 12)
        ax.set_ylim(-4.5, num_data + 1)
        ax.axis('off')

        # Zapis - transparent=True wycina tło
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.1,
                    transparent=True, dpi=300)
        plt.close(fig)
