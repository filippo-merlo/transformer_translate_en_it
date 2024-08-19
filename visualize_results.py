#%%
'''
data: 
Tokenization Level: character
Score Mean L50: 0.16250678277247846
Score Mean L100: 0.11960171970820425
Score Mean L150: 0.11557183512949935
Mean Time: 0.0037441837501263303 # for single prediction i should whatch teh whole sentence prediction


Tokenization Level: word
Score Mean L50: 0.0
Score Mean L100: 0.0
Score Mean L150: 0.0001512126623999624
Mean Time: 0.003756074685112453

'''

len([
        '<START>', ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '–','—',
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
        ':', '<', '=', '>', '?', '@', ';', 
        '[', '\\', ']', '^', '_', '`', '‘', '’', '“', '”', '…', '«', '»',
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'l', 
        'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'z', 
        'j', 'k', 'w', 'x', 'y', 
        'à', 'é', 'è', 'ì', 'ò', 'ù',
        '{', '|', '}', '~', '<PADDING>', '<END>'
    ])