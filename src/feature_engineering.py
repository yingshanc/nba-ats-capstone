import pandas as pd


def _add_rolling_point_differential(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_game_idx"] = range(len(df))
    df["_game_date"] = pd.to_datetime(df["date"])

    home_games = pd.DataFrame(
        {
            "_game_idx": df["_game_idx"],
            "season": df["season"],
            "date": df["_game_date"],
            "team": df["home"],
            "point_diff": df["score_home"] - df["score_away"],
        }
    )
    away_games = pd.DataFrame(
        {
            "_game_idx": df["_game_idx"],
            "season": df["season"],
            "date": df["_game_date"],
            "team": df["away"],
            "point_diff": df["score_away"] - df["score_home"],
        }
    )

    team_games = pd.concat([home_games, away_games], ignore_index=True)
    team_games = team_games.sort_values(["team", "season", "date", "_game_idx"])
    team_games["rolling_5_point_diff"] = team_games.groupby(
        ["team", "season"], sort=False
    )["point_diff"].transform(
        lambda s: s.shift(1).rolling(window=5, min_periods=1).mean()
    )

    rolling = team_games[["_game_idx", "team", "rolling_5_point_diff"]]
    df = df.merge(
        rolling.rename(
            columns={
                "_game_idx": "home_game_idx",
                "team": "home_team",
                "rolling_5_point_diff": "home_rolling_5_point_diff",
            }
        ),
        left_on=["_game_idx", "home"],
        right_on=["home_game_idx", "home_team"],
        how="left",
    )
    df = df.merge(
        rolling.rename(
            columns={
                "_game_idx": "away_game_idx",
                "team": "away_team",
                "rolling_5_point_diff": "away_rolling_5_point_diff",
            }
        ),
        left_on=["_game_idx", "away"],
        right_on=["away_game_idx", "away_team"],
        how="left",
    )

    home_edge = df["home_rolling_5_point_diff"] - df["away_rolling_5_point_diff"]
    df["fav_rolling_5_point_diff"] = home_edge.where(
        df["whos_favored"] == "home", -home_edge
    ).fillna(0)

    return df.drop(
        columns=[
            "_game_idx",
            "_game_date",
            "home_game_idx",
            "home_team",
            "home_rolling_5_point_diff",
            "away_game_idx",
            "away_team",
            "away_rolling_5_point_diff",
        ],
        errors="ignore",
    )


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.dropna(subset=["spread"])

    df["fav_home"] = (df["whos_favored"] == "home").astype(int)

    df["spread_abs"] = df["spread"].abs()

    df = _add_rolling_point_differential(df)

    return df
