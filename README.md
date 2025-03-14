# League of Legends Statistical Analysis

## Introduction

### Introduction to League of Legends

League of Legends (LoL) is a popular multiplayer online battle arena (MOBA) game developed and published by Riot Games. It was released in 2009 and has since become one of the most played video games in the world. The League of Legends World Championship is the largest esports event in the world by popularity, attracting millions of viewers and players worldwide.
The `league` dataset used in this analysis contains data from professional LoL matches, developed by Oracle’s Elixir. While Oracle's Elixir has been collecting data since 2014, I use data from the most recent complete season, 2024, for my analysis.

In the game, there are various monsters which when killed, provide buffs and various advantages to the team that kills them. The most important of these monsters is deemed by most players to be Baron Nashor. Baron Nashor is the most powerful neutral monster in the game that spawns in the Baron pit after 20 minutes of gameplay, and has a respawn time of 6 minutes. Killing Baron Nashor grants a powerful buff to the team that kills it, and teams often shape their late game strategies around this key objective, which plays a crucial role in determining the outcome of matches.

This analysis is centered around the question: **To what extent does getting the first Baron affect a team's performance in a match?**
I use data science techniques to analyze the impact of securing the first Baron on various match statistics as well as the overall match outcome. The culmination of this analysis is a machine learning model to predict a team's performance at the time when the first Baron spawns. This model holds the potential to be used as a real-time analysis tool for teams and analysts during matches, providing insights into the current state of the game and helping teams make informed decisions.


### The Dataset

The dataset provided by Oracle's Elixir contains detailed statistics with 161 features and 117,576 rows. I will focus on a few key features, described below:
- `gameid`: A unique identifier for each match.
- `datacompleteness`: A string indicating the completeness of the data for that match (complete or partial).
- `league`: The professional league tournament in which the match was played (e.g. LCK, LPL, LCS).
- `side`: The side of the map on which the team played (Blue or Red).
- `gamelength`: The total length of the match in seconds.
- `result`: The outcome of the match for the team (1 for a win, 0 for a loss).
- `firstbaron`: A binary indicator of whether the team secured the first Baron (1 for yes, 0 for no).
- `[stat]at20`: Various statistics at the 20-minute mark of the match, including gold, xp, cs, kills, deaths, assists. Each statistic will be explained more in detail when needed.



## Data Cleaning and Exploratory Data Analysis

### Data Cleaning

I first filtered the complete dataset to only include the features relevant to my analysis. Then I converted the `result` column to `win`, containing boolean values for whether the team won the match or not. While the dataset contained data on both player and team statistics, I focused on team statistics for my analysis and hence dropped all rows corresponding to player statistics. Since Baron Nashor spawns at 20 minutes, I also filtered the dataset to only include matches that lasted at least 20 minutes, or 1200 seconds.

There are 2778 rows where the `firstbaron` value is null. Note that all the `[stat]at20` values are also null in the rows where the `firstbaron` is null, and they are only null in these rows. These rows are indicated by a `datacompleteness` value of `partial`, which indicates that there may have been issues collecting this data for specific games. While traditional data cleaning usually also handles missing values, I held off on this step until after my assessment of the missingness of the data.

Below is the first five rows of the `league_cleaned` dataset:

| gameid             | datacompleteness   | league   | side   |   gamelength | win   |   firstbaron |   goldat20 |   xpat20 |   csat20 |   killsat20 |   assistsat20 |   deathsat20 |
|:-------------------|:-------------------|:---------|:-------|-------------:|:------|-------------:|-----------:|---------:|---------:|------------:|--------------:|-------------:|
| 10660-10660_game_1 | partial            | DCup     | Blue   |         1886 | False |          nan |        nan |      nan |      nan |         nan |           nan |          nan |
| 10660-10660_game_1 | partial            | DCup     | Red    |         1886 | True  |          nan |        nan |      nan |      nan |         nan |           nan |          nan |
| 10660-10660_game_2 | partial            | DCup     | Blue   |         1911 | False |          nan |        nan |      nan |      nan |         nan |           nan |          nan |
| 10660-10660_game_2 | partial            | DCup     | Red    |         1911 | True  |          nan |        nan |      nan |      nan |         nan |           nan |          nan |
| 10660-10660_game_3 | partial            | DCup     | Blue   |         1324 | True  |          nan |        nan |      nan |      nan |         nan |           nan |          nan |


### Univariate Analysis

I performed univariate analysis on the `gamelength` and `killsat20` features to understand their distributions.

<iframe
  src="assets/gamelength-distribution.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

The histogram above shows that the distribution of `gamelength` is roughly normal, and slightly right-skewed. This is expected, as most matches are designed to last around 25-35 minutes, but some matches can go on for longer due to various factors such as team compositions and strategies, especially in professional matches where teams are highly skilled and often evenly matched.

<iframe
  src="assets/killsat20-distribution.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

The histogram above shows the distribution of team kills at the 20-minute mark of the game. Note that while the distribution looks roughly normal and slightly right-skewed, we see an abnormally high number of 0 kills at 20 minutes. While it is unlikely that a professional team would have 0 kills this late into the game, it is possible that the data itself is messy. However, it is hard to differentiate between the two possibilities, and hence leave the data as is for now, while keeping this in mind for my analysis.


### Bivariate Analysis

I perform bivariate analysis on the `firstbaron` and `win` features to understand the relationship between whether securing the first Baron affects the outcome of the match. I use a bar plot to visualize the this relationship:

<iframe
  src="assets/firstbaron-win-plot.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

Interpreting the plot, we see that a significantly higher percentage of teams that secured the first Baron won their matches compared to those that did not. This suggests that securing the first Baron has a positive impact on a team's chances of winning the match. However, it is important to note that correlation does not imply causation, and further analysis is needed to confirm this relationship.


### Interesting Aggregates

In this section, I compute some interesting aggregate statistics:

|   firstbaron |   gamelength |      win |   goldat20 |   xpat20 |   csat20 |   killsat20 |   assistsat20 |   deathsat20 |
|-------------:|-------------:|---------:|-----------:|---------:|---------:|------------:|--------------:|-------------:|
|            0 |      1887.36 | 0.183115 |    33877.5 |  40995.2 |  685.581 |     5.3017  |       10.0476 |      7.48551 |
|            1 |      1910.35 | 0.842747 |    36075.6 |  42593.5 |  707.21  |     7.46865 |       14.3972 |      5.14195 |

Computing the mean across the columns, I get the following aggregate information:

|   firstbaron |    mean |
|-------------:|--------:|
|            0 | 6145.8  |
|            1 | 7111.55 |

This suggests that teams that secured the first Baron have a higher average across various game statistics compared to those that did not. This further supports the hypothesis that securing the first Baron has a positive impact on a team's performance in the match.



## Assessment of Missingness

### NMAR Analysis

In the original dataset, I noticed that the `ban1`, `ban2`, `ban3`, `ban4`, and `ban5` columns contained a significant number of null values with no discernible pattern. I strongly believe that this column is NMAR (Not Missing At Random). This is because before every League of Legends game starts, players are allowed to ban certain champions from being played in the game. I believe that this data is NMAR because a player may choose not to ban any champion, and hence the missingness of this data depends itself.

To determine whether this data is instead MAR (Missing At Random), an additional data I would need would be a boolean column indicating whether a player chose to ban a champion or not.


### Missingness Dependency

As mentioned previously, many rows in the `league_cleaned` dataset have `NaN` values for the `firstbaron` and `[stat]at20` columns, indicated by a `datacompleteness` value of `partial`. In this section, I try to determine whether the missingness of the `firstbaron` column depends on other columns. In particular, the columns I will use to determine this are the `league` and `side` columns. For both cases, I use a permutation test with a 1% significance level to determine whether the missingness of the `firstbaron` column depends on the `league` and `side` columns.

First, I perform a permutation test on the `league` column.

**Null Hypothesis**: The distribution of `league` when `firstbaron` is missing is the same as the distribution of `league` when `firstbaron` is not missing.

**Alternative Hypothesis**: The distribution of `league` when `firstbaron` is missing is not the same as the distribution of `league` when `firstbaron` is not missing.

**Test Statistic**: Total Variation Distance (TVD). This is appropriate because we are comparing two categorical distributions.

**Significance Level**: 1%

Below is the observed distribution of the `league` column when `firstbaron` is missing and not missing:

| league          |   firstbaron_missing = False |   firstbaron_missing = True |
|:----------------|-----------------------------:|----------------------------:|
| AC              |                   0.0041836  |                  0          |
| AL              |                   0.0181688  |                  0          |
| CBLOL           |                   0.0314368  |                  0          |
| CBLOLA          |                   0.0329907  |                  0          |
| CDF             |                   0.00788907 |                  0          |
| CT              |                   0.00454219 |                  0          |
| DCup            |                 nan          |                  0.012959   |
| EBL             |                   0.0169735  |                  0          |
| EBLPA           |                   0.00298829 |                  0          |
| EM              |                   0.047693   |                  0          |
| EPL             |                   0.0178102  |                  0          |
| ESLOL           |                   0.034186   |                  0          |
| EWC             |                   0.0022711  |                  0          |
| GLL             |                   0.0181688  |                  0          |
| GLLPA           |                   0.00454219 |                  0          |
| HC              |                   0.013268   |                  0          |
| HM              |                   0.0185274  |                  0          |
| HW              |                   0.00968205 |                  0          |
| IC              |                   0.00788907 |                  0          |
| KeSPA           |                   0.00585704 |                  0          |
| LAS             |                   0.0344251  |                  0          |
| LCK             |                   0.0576142  |                  0          |
| LCKC            |                   0.060961   |                  0          |
| LCO             |                   0.0181688  |                  0          |
| LCS             |                   0.02295    |                  0          |
| LDL             |                 nan          |                  0.406048   |
| LEC             |                   0.0351422  |                  0          |
| LFL             |                   0.028329   |                  0          |
| LFL2            |                   0.0203203  |                  0          |
| LIT             |                   0.0175711  |                  0          |
| LJL             |                   0.0176907  |                  0          |
| LLA             |                   0.0253407  |                  0          |
| LPL             |                 nan          |                  0.516199   |
| LPLOL           |                   0.0182883  |                  0          |
| LRN             |                   0.0105188  |                  0          |
| LRS             |                   0.0121922  |                  0          |
| LVP SL          |                   0.0289266  |                  0          |
| MSI             |                 nan          |                  0.0554356  |
| NACL            |                   0.0583313  |                  0          |
| NEXO            |                   0.019125   |                  0          |
| NLC             |                   0.024743   |                  0          |
| NLC Aurora Open |                   0.00920392 |                  0          |
| PCS             |                   0.0352618  |                  0          |
| PRM             |                   0.0288071  |                  0          |
| PRMP            |                   0.0154196  |                  0          |
| TCL             |                   0.0216352  |                  0          |
| TSC             |                   0.0121922  |                  0          |
| UL              |                   0.0190055  |                  0          |
| USP             |                   0.00454219 |                  0          |
| VCS             |                   0.0300024  |                  0          |
| WLDs            |                   0.0142242  |                  0.00935925 |

After performing the permutation test, I computed an observed test statistic of 0.4953203743700504 and got a p-value of 0.0. Below is a plot of the empirical distribution of the test statistic from the permutation test, along with the observed test statistic:

<iframe
  src="assets/league-simulation.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

Hence, I reject the null hypothesis and conclude that the distribution of `league` when `firstbaron` is missing is not the same as the distribution of `league` when `firstbaron` is not missing. This suggests that the missingness of the `firstbaron` column is MAR on the `league` column.


Next, I perform a permutation test on the `side` column.

**Null Hypothesis**: The distribution of `side` when `firstbaron` is missing is the same as the distribution of `side` when `firstbaron` is not missing.

**Alternative Hypothesis**: The distribution of `side` when `firstbaron` is missing is not the same as the distribution of `side` when `firstbaron` is not missing.

**Test Statistic**: Total Variation Distance (TVD). This is appropriate because we are comparing two categorical distributions.

**Significance Level**: 1%

The observed distribution of the `side` column when `firstbaron` is missing and not missing is as follows:

| side   |   firstbaron_missing = False |   firstbaron_missing = True |
|:-------|-----------------------------:|----------------------------:|
| Blue   |                          0.5 |                         0.5 |
| Red    |                          0.5 |                         0.5 |

In this case, I computed an observed test statistic of 0.0 and got a p-value of 1.0. Below is a plot of the empirical distribution of the test statistic from the permutation test, along with the observed test statistic:

<iframe
  src="assets/side-simulation.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

Hence, I fail to reject the null hypothesis and conclude that the distribution of `side` when `firstbaron` is missing is the same as the distribution of `side` when `firstbaron` is not missing. This suggests that the missingness of the `firstbaron` column is not dependent on the `side` column.


### Handling Missing Values

Now that I have completed my assessment of the missingness of the data, I can handle the missing values in the `firstbaron` and `[stat]at20` columns. Counting the nulls, I find that there are 2778 out of 19510 rows with null values. Since I cannot impute these values (no way to know if a team got the first baron or not), I drop these rows by dropping all rows with a `datacompleteness` value of `partial`. I also drop the `datacompleteness` column, as it is no longer needed. Lastly, without any missing values, I convert the `firstbaron` and `[stat]at20` columns from floats to integers as that is the appropriate data type for these columns.

Below is the first five rows of the `league_cleaned` dataset after handling the missing values:

| gameid           | league   | side   |   gamelength | win   |   firstbaron |   goldat20 |   xpat20 |   csat20 |   killsat20 |   assistsat20 |   deathsat20 |
|:-----------------|:---------|:-------|-------------:|:------|-------------:|-----------:|---------:|---------:|------------:|--------------:|-------------:|
| LOLTMNT99_132542 | TSC      | Blue   |         1446 | True  |            1 |      38246 |    42428 |      718 |          10 |            21 |            5 |
| LOLTMNT99_132542 | TSC      | Red    |         1446 | False |            0 |      33998 |    40290 |      668 |           5 |            10 |           10 |
| LOLTMNT99_132665 | TSC      | Blue   |         2122 | True  |            0 |      36104 |    43211 |      701 |          11 |            17 |            8 |
| LOLTMNT99_132665 | TSC      | Red    |         2122 | False |            1 |      35327 |    40489 |      677 |           8 |            11 |           11 |
| LOLTMNT99_132755 | TSC      | Blue   |         2099 | True  |            1 |      33386 |    42148 |      666 |           7 |             9 |            7 |



## Hypothesis Testing

