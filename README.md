# Branch for keeping logs from Inspect

# Important notes

- .gitignore ignores everything besides logs directory and relevant files for serving logs
- Some logs could be very large, so `git lfs` is used

# Logs description
- `logs_gh_pages` - bundled version of logs suitable to serve via github pages
- `logs_raw` - Logs output after running evals. Used for `logs_gh_pages` creation

# How to add new logs
0. Instal [git lfs](https://git-lfs.com)

1. Put files inside `logs_raw`. If necessary create new subdirectory with logs

2. Run following command to overwrite existing logs in `logs_gh_pages` and add new logs
    ```
    inspect view bundle --log-dir logs_raw --output-dir logs_gh_pages --overwrite
    ```
3. Add commit and push changes to github
 
# How to view the logs
- Go [here](https://diogo-cruz.github.io/algoverse_mountain/logs_gh_pages/)
