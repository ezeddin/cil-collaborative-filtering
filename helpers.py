
class Logger:
    counter = 0
    sieve = 0
    
    def __init__(self,sieve=0):
        self.sieve = sieve
        
    def reset(self):
        self.counter = 0
    
    def log(self, message):
        if(self.sieve == 0):
            print(message)
        else:
            self.counter += 1
            if (self.counter % self.sieve == 0):
                print(message)
                self.reset()
            
    