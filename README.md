# Bradley-Terry Model Ranking Script

A small script to compute Bradley–Terry model that:

1. Pulls down pairwise match data from a Google spreadsheet.
2. Computes either a point estimate or a full-Bayesian estimate of the
   Bradley-Terry model with a simple regularization scheme (dummy games).
3. Uploads the scaled scores back to the same Google spreadsheet in a new tab.

I quickly wrote this up in a (couple of) afternoons so that we could track our
relative ratings in table tennis at the office.  I put it on a cron job and
give everyone access to the sheet at work.  Check out the post I wrote about it:
[Building A Table Tennis Ranking Model](https://rubikloud.com/lab/building-a-table-tennis-ranking-model/)

## Usage

    usage: update_model.py [-h] --creds-file CREDS_FILE [--sheet-key SHEET_KEY]
                           [--backup-dir BACKUP_DIR] [--model MODEL]
                           [--data-sheet DATA_SHEET] [--rank-sheet RANK_SHEET]
                           [--num-samples NUM_SAMPLES] [--num-tune NUM_TUNE]
                           [--interval INTERVAL] [--alpha ALPHA]
    
    Script to update Google Sheet with ranking model
    
    optional arguments:
      -h, --help            show this help message and exit
      --creds-file CREDS_FILE
                            Google Drive API sheet
      --sheet-key SHEET_KEY
                            Google sheet key from URL
      --backup-dir BACKUP_DIR
                            dir to backup game data
      --model MODEL         'point' for point estimate model, 'bayes' for full
                            bayesian model
      --data-sheet DATA_SHEET
                            Name of worksheet containing game data
      --rank-sheet RANK_SHEET
                            Name of worksheet to write rankings
      --num-samples NUM_SAMPLES
                            Number of samples to draw for HMC
      --num-tune NUM_TUNE   Number of tuning samples to draw for HMC
      --interval INTERVAL   Size of interval for 'bayes' model (%)
      --alpha ALPHA         Regularization parameter

## Notes

 * You will need to setup an API key for gspread to access your Google sheet: http://gspread.readthedocs.io/en/latest/oauth2.html
 * Make sure you share the target Google sheet with the email address associated with your credentials (usually a randomly generated one).
 * The "Game Data" sheet should have 5 columns: `date`, `Player A`, `Player B`, `Wins A`, `Wins B`
 * The "Ranking" sheet show the score for each player scaled between 0 and 1000 on a log scale for the "point" model.  
 * The ranking can be interpreted as probability of Player A beating Player B is (Score_A) / (Score_A + Score_B), if you convert back to a non-log scale.
 * The "bayes" model will add two columns for the posterior credible interval.
 * The interval can be interpreted as "the (interval)% probability that the score is between these the low and high score" centered on the posterior mode.
