import logging
import pandas as pd
import numpy as np
from utils import *


class CrapsRunner:
    """
    Simulation of the game craps
    """
    def __init__(self,
                 bankroll: int = 10000,
                 num_rolls: int = 100000,
                 min_bet: int = 25):
        """
        initializing the craps runner
        :param funds: int
            the starting amount of funds to play
        :param min_bet:
            the minimum bet that can be played
        """
        self.bankroll = bankroll
        self.num_rolls = num_rolls
        self.min_bet = min_bet
        self.df_roll_outcomes = self.calc_roll_outcome_odds()

    @staticmethod
    def get_place_bet_payoff(x: int) -> float:
        """

        :param x:
        :return:
        """
        if x in [4, 10]:
            return 9/5
        elif x in [5, 9]:
            return 7/5
        elif x in [6, 8]:
            return 7/6
        else:
            return np.nan



    @staticmethod
    def get_odds_bet_payoff(x: int) -> float:
        """

        :param x:
        :return:
        """
        if x in [4, 10]:
            return 2/1
        elif x in [5, 9]:
            return 3/2
        elif x in [6, 8]:
            return 6/5
        else:
            return np.nan

    @staticmethod
    def get_dice_sum_probability(k: int) -> float:
        """
        Get the probability for the sum of a dice roll
        :param k: int
            sum of dice roll
        :return: float
            probability of summing dice roll
        """
        return (6 - (abs(k - 7))) / 36

    @staticmethod
    def get_expected_num_rolls(p: int) -> float:
        """
        Get the expected number of rolls for the sum of a dice roll
        :param p:
        :return:
        """
        return 1 / p


    def calc_roll_outcome_odds(self) -> pd.DataFrame:
        """

        :return:
        """
        roll_outcome_odds = [x for x in range(2, 13)]
        df = pd.DataFrame(roll_outcome_odds, columns=['Roll'])
        df["Odds"] = df["Roll"].apply(lambda x: self.get_dice_sum_probability(k=x))
        df["Expected_Rolls"] = df["Odds"].apply(lambda p: self.get_expected_num_rolls(p=p))
        df["Place_Bet_Payoff"] = df["Roll"].apply(lambda x: self.get_place_bet_payoff(x=x))
        df["Odds_Bet_Payoff"] = df["Roll"].apply(lambda x: self.get_odds_bet_payoff(x=x))
        return df


    def get_payout_earnings(self, outcome: int, bet_amount: int, bet_play: str) -> float:
        """

        :param outcome:
        :param bet_amount:
        :param bet_play:
        :return:
        """
        if bet_play not in ["place_bet", "odds_bet", "pass_line"]:
            raise ValueError("Invalid bet play")
        # get the payout earnings
        if bet_play == "pass_line":
            return bet_amount
        elif bet_play == "odds_bet":
            payout_odds = self.df_roll_outcomes[self.df_roll_outcomes["Roll"] == outcome]["Odds_Bet_Payoff"].values[0]
        else:
            payout_odds = self.df_roll_outcomes[self.df_roll_outcomes["Roll"] == outcome]["Place_Bet_Payoff"].values[0]
        return payout_odds * bet_amount

    @staticmethod
    def occurence_counter(sample_rolls: list,
                          point: int,
                          place_bet: int | list) -> list:
        """

        :param sample_rolls:
        :param point:
        :param place_bet:
        :return:
        """
        occurence_list = []
        counter = 0
        for x in sample_rolls:
            if isinstance(place_bet, list):
                if x in place_bet:
                    counter += 1
            if isinstance(place_bet, int):
                if x == place_bet:
                    counter += 1
            if x == point or x == 7:
                occurence_list.append(counter)
                counter = 0
        return occurence_list


    def simulate_place_bet_hits(self,
                                point: int,
                                place_bet: int | list,
                                num_rolls: int = 100000) -> list:
        """

        :param point:
        :param place_bet:
        :param num_rolls:
        :return:
        """
        probs = self.df_roll_outcomes["Odds"].values
        rolls = self.df_roll_outcomes["Roll"].values
        # Generate the samples for dice rolls
        sample_rolls = np.random.choice(rolls, num_rolls, p=probs)
        logging.info(sample_rolls)
        return self.occurence_counter(sample_rolls, point, place_bet)


    # def game_simulation(self, ):


if __name__ == '__main__':
    # initialize the logger
    init_logger(file_name="craps_runner.log")
    craps_runner = CrapsRunner(bankroll=10000, num_rolls=100000)
    logging.info(craps_runner.df_roll_outcomes)
    logging.info(craps_runner.df_roll_outcomes.Odds.sum())
    logging.info(craps_runner.simulate_place_bet_hits(point=4, place_bet=[6, 8], num_rolls=10))


