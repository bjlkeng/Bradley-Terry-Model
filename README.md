# Bradley-Terry Model Ranking Script

A small script to compute Bradley–Terry model that:

1. Pulls down pairwise match data from a Google spreadsheet.
2. Computes the Bradley-Terry model with a simple regularization scheme (dummy games).
3. Uploads the scaled scores back to the same Google spreadsheet in a new tab.

I quickly wrote this up in an afternoon so that we could track our relative
rankings in ping ping at the office.  I put it on a cron job and give everyone
access to the sheet at work.

# Notes

 * You will need to setup an API key for gspread to access your Google sheet: http://gspread.readthedocs.io/en/latest/oauth2.html
 * Make sure you share the target Google sheet with the email address associated with your credentials (usually a randomly generated one).
 * The "Game Data" sheet should have 5 columns: `date`, `Player A`, `Player B`, `Wins A`, `Wins B`
 * The "Ranking" sheet show the score for each player scaled between 0 and 1000.  It can be interpreted as probability of Player A beating Player B is (Score_A) / (Score_A + Score_B).
