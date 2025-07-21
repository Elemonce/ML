import random

class BirthdayParadox:
    def __init__(self, trials=1000):
        self.trials = trials


    def start(self):
        hist = []
        loops = 0
        while loops < self.trials:
            attempts = 0
            birthday_days = []
            while True:
                attempts += 1  
                random_day = random.choice(range(1, 367))
                if random_day in birthday_days:
                    hist.append(attempts)
                    loops += 1
                    break
                birthday_days.append(random_day)
        
        mean_attempts = sum(hist) / len(hist)

        print(f"With {self.trials} trials, on average, two people with the same birthday were found after {mean_attempts} attempts.")
                


birthday = BirthdayParadox()

birthday.start()
            
