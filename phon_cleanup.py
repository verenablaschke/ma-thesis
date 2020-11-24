

DATA_DIR = './data/'
IN_FILE = DATA_DIR + 'phon_parsed.tsv'
OUT_FILE = DATA_DIR + 'phon_cleaned.tsv'


with open(IN_FILE, 'r', encoding='utf8') as in_file:
    with open(OUT_FILE, 'w', encoding='utf8') as out_file:
        for line in in_file:
            cells = line.split('\t')
            utterance = cells[-1]
            utterance = utterance.replace('nng', 'ŋŋ')
            utterance = utterance.replace('ng', 'ŋ')
            utterance = utterance.replace('\'ŋ', 'ŋ̍')  # TODO WILL THIS BE PARSED AS TWO LETTERS LATER?
            utterance = utterance.replace('\'m', 'm̩')
            utterance = utterance.replace('\'n', 'n̩')
            utterance = utterance.replace('\'l', 'l̩')
            utterance = utterance.replace('\'r', 'r̩')
            utterance = utterance.replace('\'s', 's̩')
            utterance = utterance.replace('\'L', ' ̍ɽ')
            utterance = utterance.replace('L', 'ɽ')
            utterance = utterance.replace('\'rn', 'ɳ̩')  # TODO the only ɳ in the data????
            utterance = utterance.replace('kkj', 'çç')
            utterance = utterance.replace('kj', 'ç')
            utterance = utterance.replace('ssj', 'ʂʂ')  # TODO ambiguous????
            utterance = utterance.replace('sj', 'ʂ')
            utterance = utterance.replace('ttj', 'tʧ') 
            utterance = utterance.replace('tj', 'ʧ') 
            utterance = utterance.replace('\'i', '.i') 
            utterance = utterance.replace('\'e', '.e')
            utterance = utterance.replace('\'u', '.u')
            utterance = utterance.replace('\'æ', '.æ')
            utterance = utterance.replace('\'o', '.o')
            utterance = utterance.replace('\'y', '.y')
            utterance = utterance.replace('\'a', '.a')

            out_file.write('\t'.join(cells[0:-1]))
            out_file.write('\t')
            out_file.write(utterance)

            

            
            