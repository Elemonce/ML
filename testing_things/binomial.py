import random

class GetYourDie:
    def __init__(self, numbers_to_attempt=1000, trials=4, times_to_get_die=2):
        self.numbers_to_attempt = numbers_to_attempt
        self.trials = trials
        self.times_to_get_die = times_to_get_die

    def start(self):
        number = int(input("Enter your number: "))
        attempted_times = 0

        hist = []
        die_values = [1, 2, 3, 4, 5, 6]
        while attempted_times < self.numbers_to_attempt:
            times_got = 0
            for _ in range(self.trials):
                random_die = random.choice(die_values)
                if random_die == number:
                    times_got += 1
            if times_got >= self.times_to_get_die:
                hist.append(1)
            else:
                hist.append(0)
            attempted_times += 1

        
        successful_attempts = sum(hist)
        print(f"Out of {self.numbers_to_attempt} attempts, {successful_attempts} attempts were successful.")
        print(f"Success rate: {successful_attempts / self.numbers_to_attempt}")



binomial = GetYourDie()

binomial.start()
