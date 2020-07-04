class Hangman:
    def __init__(self, name, word, numGo):
        self.name = name
        self.word = word
        self.numGo = numGo

    def play(self):
        trys = self.numGo
        corrects = 0
        separate = []
        for char in self.word:
            separate.append(char)
        lines2 = ['_'] * len(self.word)
        print (lines2)
        allLetters = "abcdefghijklmnopqrstuvwxyz"
        while self.numGo > 0:
            choice = input("choose a letter: ")
            if choice in (allLetters):
                allLetters = allLetters.replace(choice, '_', 1)
                print (f"remaining letters: {allLetters}")
                if choice in separate:
                    letters = (list(i for i, e in enumerate(separate) if e == choice))
                    for letter in letters:
                        lines2[letter] = choice
                        corrects += 1
                    print(lines2)
                    if corrects == len(self.word):
                        print (f"congratulations {self.name}, you correctly guessed {self.word}, go have a beer")
                        break
                else:
                    self.numGo -= 1
                    print (f"{self.numGo} attempts left...")
        if self.numGo == 0:
            print ("If you survived the fall without breaking your neck, you have slowly choked to death")
            print (f"all because you couldnt guess the word {self.word}")







hanging = Hangman('Adele and Freddie', 'spandex', 6)
hanging.play()
