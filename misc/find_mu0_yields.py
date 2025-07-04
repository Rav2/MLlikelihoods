import pandas as pd
import sys

def find_closest_row(csv_path):
    try:
        # Wczytaj plik CSV
        df = pd.read_csv(csv_path)

        # Sprawdź, czy plik ma wystarczającą liczbę kolumn
        if df.shape[1] < 6:
            print("Plik CSV musi zawierać przynajmniej 6 kolumn.")
            return

        # Oblicz różnicę między wartościami w -5 i -6 kolumnie
        df['abs_diff'] = (df.iloc[:, -5] - df.iloc[:, -6]).abs()

        # Znajdź indeks wiersza z najmniejszą różnicą
        closest_row_idx = df['abs_diff'].idxmin()

        # Wyświetl wiersz z najmniejszą różnicą
        print("Najbliższy wiersz:")
        print(df.iloc[closest_row_idx])
        print(list(df.iloc[closest_row_idx]))


    except FileNotFoundError:
        print(f"Nie znaleziono pliku: {csv_path}")
    except Exception as e:
        print(f"Wystąpił błąd: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Użycie: python skrypt.py sciezka_do_csv")
    else:
        csv_path = sys.argv[1]
        find_closest_row(csv_path)
