import argparse
import gspread
import logging
import numpy as np
import os
import pandas as pd
import time

from datetime import datetime
from collections import Counter
from oauth2client.service_account import ServiceAccountCredentials

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

DUMMY_PLAYER = 'DUMMY PLAYER'


def connect_sheet(creds_file, sheet_key):
    scope = ['https://spreadsheets.google.com/feeds']
    credentials = ServiceAccountCredentials.from_json_keyfile_name(creds_file, scope)
    gc = gspread.authorize(credentials)

    return gc.open_by_key(sheet_key)


def extract_game_data(sheet, sheet_name):
    worksheet = sheet.worksheet(sheet_name)
    rawdata = worksheet.get_all_values()

    df = pd.DataFrame(rawdata[1:], columns=rawdata[0])
    assert all(c in df.columns for c in ['Date', 'Player A', 'Player B', 'Wins A', 'Wins B']), \
        'Expecting columns Date, Player A, Player B, Wins A, Wins B'

    df['Date'] = df['Date'].astype(datetime)
    df['Wins A'] = df['Wins A'].astype(int)
    df['Wins B'] = df['Wins B'].astype(int)

    return df


def add_dummy_games(game_data, alpha=1):
    ''' Regularizes the estimate by adding games against a dummy player.

        :param alpha: regularization parameter, number dummy wins/loses to add
    '''
    players = sorted(list(set(game_data['Player A']) | set(game_data['Player B'])))

    # Add dummy games
    dummy_data = [[datetime(2000, 1, 1), p, DUMMY_PLAYER, alpha, alpha] for p in players]
    df = pd.DataFrame(dummy_data, columns=game_data.columns)
    df = pd.concat([game_data, df])
    df

    return df


def compute_rank_scores(game_data, max_iters=1000, error_tol=1e-3):
    ''' Computes Bradley-Terry using iterative algorithm

        See: https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model
    '''
    # Do some aggregations for convenience
    # Total wins per player
    winsA = game_data.groupby('Player A').agg(sum)['Wins A'].reset_index()
    winsA = winsA[winsA['Wins A'] > 0]
    winsA.columns = ['Player', 'Wins']
    winsB = game_data.groupby('Player B').agg(sum)['Wins B'].reset_index()
    winsB = winsB[winsB['Wins B'] > 0]
    winsB.columns = ['Player', 'Wins']
    wins = pd.concat([winsA, winsB]).groupby('Player').agg(sum)['Wins']

    # Total games played between pairs
    num_games = Counter()
    for index, row in game_data.iterrows():
        key = tuple(sorted([row['Player A'], row['Player B']]))
        total = sum([row['Wins A'], row['Wins B']])
        num_games[key] += total

    # Iteratively update 'ranks' scores
    players = sorted(list(set(game_data['Player A']) | set(game_data['Player B'])))
    ranks = pd.Series(np.ones(len(players)) / len(players), index=players)
    for iters in range(max_iters):
        oldranks = ranks.copy()
        for player in ranks.index:
            denom = np.sum(num_games[tuple(sorted([player, p]))]
                           / (ranks[p] + ranks[player])
                           for p in ranks.index if p != player)
            ranks[player] = 1.0 * wins[player] / denom

        ranks /= sum(ranks)

        if np.sum((ranks - oldranks).abs()) < error_tol:
            break

    if np.sum((ranks - oldranks).abs()) < error_tol:
        logging.info(" * Converged after %d iterations.", iters)
    else:
        logging.info(" * Max iterations reached (%d iters).", max_iters)

    del ranks[DUMMY_PLAYER]

    # Scale score to be between 1 and 1000
    ranks = (1000 * ranks.sort_values(ascending=False)).astype(int).clip(1)

    return ranks


def upload_to_gsheets(sheet, ranks, rank_sheet):
    ranking_ws = [w for w in sheet.worksheets() if w.title == rank_sheet]
    if not ranking_ws:
        worksheet = sheet.add_worksheet(title=rank_sheet, rows=100, cols=25)
    else:
        worksheet = ranking_ws[0]

    worksheet.clear()
    worksheet.update_cell(1, 1, 'Player')
    worksheet.update_cell(1, 2, 'Score')

    if worksheet.row_count < len(ranks) + 1:
        worksheet.resize(rows=len(ranks) + 1, cols=25)

    player_cells = worksheet.range(2, 1, len(ranks) + 1, 1)
    for cell, player in zip(player_cells, ranks.index):
        cell.value = player

    value_cells = worksheet.range(2, 2, len(ranks) + 1, 2)
    for cell, val in zip(value_cells, ranks.values):
        cell.value = val

    worksheet.update_cells(player_cells + value_cells)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to update Google Sheet with ranking model")
    parser.add_argument('--creds-file', type=str, required=True,
                        help='Google Drive API sheet')
    parser.add_argument('--sheet-key', default=None, type=str, help='Google sheet key from URL')
    parser.add_argument('--backup-dir', default=None, type=str,
                        help='dir to backup game data')
    parser.add_argument('--data-sheet', default='Game Data', type=str,
                        help='Name of worksheet containing game data')
    parser.add_argument('--rank-sheet', default='Ranking', type=str,
                        help='Name of worksheet to write rankings')
    parser.add_argument('--alpha', default=1, type=int, help='Regularization parameter')
    args = parser.parse_args()

    logging.info("Connecting to Google sheets '%s'...", args.sheet_key)
    sheet = connect_sheet(args.creds_file, args.sheet_key)

    logging.info("Extracting game data from '%s'...", args.data_sheet)
    game_data = extract_game_data(sheet, args.data_sheet)
    logging.info(" * Found %d rows.", len(game_data))

    if args.backup_dir:
        filename = "game_data-%d.csv.gz" % int(time.time())
        logging.info(" * Saving backup to '%s'", filename)
        game_data.to_csv(os.path.join(args.backup_dir, filename),
                         compression='gzip', index=False)

    logging.info("Adding dummy game for regularization (alpha=%d)...", args.alpha)
    game_data = add_dummy_games(game_data, args.alpha)

    logging.info("Computing rank scores...")
    ranks = compute_rank_scores(game_data)

    logging.info("Uploading rank scores to sheet '%s'...", args.rank_sheet)
    upload_to_gsheets(sheet, ranks, args.rank_sheet)

    logging.info("Done")
