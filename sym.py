import re
import math
import numpy as np
from difflib import get_close_matches

def map_futures_symbol(symbol: str) -> str:
    if symbol is None or (isinstance(symbol, float) and math.isnan(symbol)):
        return None

    symbol = symbol.strip().upper()

    # Remove extra spaces and suffixes like -1, -3, .3, -100T
    symbol = re.sub(r"[\s\-\.]\d*[A-Z]*\d*$", "", symbol)

    # If it starts with "/", just remove the slash
    if symbol.startswith("/"):
        symbol = symbol[1:]
    
    if symbol.startswith("6") and "6J" not in symbol:
        symbol = symbol.replace("6","G")

    # Micro/Mini mappings
    micro_map = {
        "MNQ": "MNQ",
        "MES": "MES",
        "MCL": "MCL",
        "MGC": "MGC",
        "MBT": "MBT",
        "MHG": "MHG",
        "MNG": "MNG",
        "FDX": "FDX",
        "SIL":"SIL"
    }

    base_map = {
        "ES": "ES",
        "NQ": "NQ",
        "YM": "YM",
        "CL": "CL",
        "GC": "GC",
        "HG": "HG",
        "NG": "NG",
        "PL": "PL",
        "RB": "RB",
        "ZB": "ZB",
        "ZF": "ZF",
        "6J": "6J",
        "6B": "6B",
        "NKD": "NKD",
        "RTY": "RTY",
        "CPE": "CPE",
        "SI":"SI"
    }

    # 1️⃣ Exact micro match
    for micro, normal in micro_map.items():
        if symbol.startswith(micro):
            return normal,True

    # 2️⃣ Exact base match
    cleaned = symbol
    for base, normal in base_map.items():
        if cleaned.startswith(base):
            return normal,False

    return cleaned,False

if __name__ == "__main__":
    
    symbols = ['YMU25', 'METN25', 'HG', 'GCQ25', 'YM', 'MNQ', 'MNQM25', 'NQM25',
            'MGCQ25', 'GCQ23', 'MCLQ25', np.nan, 'MBTN25', 'CL', 'CLQ25',
            'NQU25', 'GC M25', '6JM25', 'NQ', 'NQM25-3', 'GCM25', 'ESM25',
            'GC', 'GCZ25', '/GC', 'CLU25', 'XAUUSD', 'MNQU25', 'MGC225',
            'MGCZ25', 'MNQH25', 'MHGN25', 'MGCJ25', 'GCJ25', 'MGC', '6BU25',
            '/HG', 'MNGK25', 'MHGK25', 'MGCM25', 'HGK25', 'GCZ225', 'NKDH25',
            'PLV25-3', 'MNQM25.3', 'PLV25', 'CLJ25', 'MNQH25.5', 'CL25',
            'MBTJ25', 'MBT', 'HGU25', 'NGU25', 'NQ H25', 'NQH25', 'MGC25',
            'NKDM25', 'ESH25', 'CLN25', 'NQM25-100T', 'NQ M25', 'MESM25',
            'PLJ25', 'FDXM25', '/ES', 'PLN25', 'NKD', 'NKDU25', 'HGN25',
            'YMM25', 'FDXM JUN25', 'HGN5', 'ZFM25', 'MHGU25', 'NQU25-5',
            '6JU25', 'ESU25', '/YM', 'YMU25-1', 'RTYU25', 'MCL K25', 'MCL25',
            'CPEN25', 'RBH25', 'RB25', 'RB H25', 'CLH25', 'MCLM25', 'MCL',
            'CLM25', '/CL', 'GC G25', 'GCG25', 'MGCG25', 'MGC G25', 'NGM25',
            '/NQ', 'GCQ2025', 'HG U25', 'GCQ25-1']

    mapped = [map_futures_symbol(s) for s in symbols]
    print(mapped)

