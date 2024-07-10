hotpot_qa_train_content = """
Example({'question': 'At My Window was released by which American singer-songwriter?', 'answer': 'John Townes Van Zandt'}) (input_keys={'question'}),
 Example({'question': 'which  American actor was Candace Kita  guest starred with ', 'answer': 'Bill Murray'}) (input_keys={'question'}),
 Example({'question': 'Which of these publications was most recently published, Who Put the Bomp or Self?', 'answer': 'Self'}) (input_keys={'question'}),
 Example({'question': 'The Victorians - Their Story In Pictures is a documentary series written by an author born in what year?', 'answer': '1950'}) (input_keys={'question'}),
 Example({'question': 'Which magazine has published articles by Scott Shaw, Tae Kwon Do Times or Southwest Art?', 'answer': 'Tae Kwon Do Times'}) (input_keys={'question'}),
 Example({'question': 'In what year was the club founded that played Manchester City in the 1972 FA Charity Shield', 'answer': '1874'}) (input_keys={'question'}),
 Example({'question': 'Which is taller, the Empire State Building or the Bank of America Tower?', 'answer': 'The Empire State Building'}) (input_keys={'question'}),
 Example({'question': 'Which American actress who made their film debut in the 1995 teen drama "Kids" was the co-founder of Voto Latino?', 'answer': 'Rosario Dawson'}) (input_keys={'question'}),
 Example({'question': 'Tombstone stared an actor born May 17, 1955 known as who?', 'answer': 'Bill Paxton'}) (input_keys={'question'}),
 Example({'question': 'What is the code name for the German offensive that started this Second World War engagement on the Eastern Front (a few hundred kilometers from Moscow) between Soviet and German forces, which included 102nd Infantry Division?', 'answer': 'Operation Citadel'}) (input_keys={'question'}),
 Example({'question': 'Who acted in the shot film The Shore and is also the youngest actress ever to play Ophelia in a Royal Shakespeare Company production of "Hamlet." ?', 'answer': 'Kerry Condon'}) (input_keys={'question'}),
 Example({'question': 'Which company distributed this 1977 American animated film produced by Walt Disney Productions for which Sherman Brothers wrote songs?', 'answer': 'Buena Vista Distribution'}) (input_keys={'question'}),
 Example({'question': 'Samantha Cristoforetti and Mark Shuttleworth are both best known for being first in their field to go where? ', 'answer': 'space'}) (input_keys={'question'}),
 Example({'question': 'Having the combination of excellent foot speed and bat speed helped Eric Davis, create what kind of outfield for the Los Angeles Dodgers? ', 'answer': '"Outfield of Dreams"'}) (input_keys={'question'}),
 Example({'question': 'Which Pakistani cricket umpire who won 3 consecutive ICC umpire of the year awards in 2009, 2010, and 2011 will be in the ICC World Twenty20?', 'answer': 'Aleem Sarwar Dar'}) (input_keys={'question'}),
 Example({'question': 'The Organisation that allows a community to influence their operation or use and to enjoy the benefits arisingwas founded in what year?', 'answer': '2010'}) (input_keys={'question'}),
 Example({'question': '"Everything Has Changed" is a song from an album released under which record label ?', 'answer': 'Big Machine Records'}) (input_keys={'question'}),
 Example({'question': 'Who is older, Aleksandr Danilovich Aleksandrov or Anatoly Fomenko?', 'answer': 'Aleksandr Danilovich Aleksandrov'}) (input_keys={'question'}),
 Example({'question': 'On the coast of what ocean is the birthplace of Diogal Sakho?', 'answer': 'Atlantic'}) (input_keys={'question'}),
 Example({'question': 'This American guitarist best known for her work with the Iron Maidens is an ancestor of a composer who was known as what?', 'answer': 'The Waltz King'}) (input_keys={'question'})
"""
custom_trainset_content="""
Example({'question': '1+1=?', 'answer': '2'}) (input_keys={'question'}),
Example({'question': '1+5=?', 'answer': '6'}) (input_keys={'question'}),
Example({'question': '3+3=?', 'answer': '6'}) (input_keys={'question'}),
Example({'question': '5*5=?', 'answer': '25'}) (input_keys={'question'}),
Example({'question': '6+6=?', 'answer': '12'}) (input_keys={'question'}),
"""

custom_testset_content="""
Example({'question': '1+1+5=?', 'answer': '7'}) (input_keys={'question'}),
Example({'question': '1+5+5=?', 'answer': '11'}) (input_keys={'question'}),
Example({'question': '3+3+5=?', 'answer': '11'}) (input_keys={'question'}),
Example({'question': '5*5+5=?', 'answer': '30'}) (input_keys={'question'}),
Example({'question': '6+6+5=?', 'answer': '17'}) (input_keys={'question'}),
"""

import ast
import dspy

def get_dataset_from_str(content, has_gold_reason=True):
    split_content = content.split("Example(")[1:]
    questions = []
    answers = []
    if has_gold_reason:
        gold_reasonings = []

    for item in split_content:
        start_index = item.index("{")
        end_index = item.index("}")
        dict_str = item[start_index:end_index + 1]

        parsed_dict = ast.literal_eval(dict_str)
        questions.append(parsed_dict['question'])
        answers.append(parsed_dict['answer'])
        if has_gold_reason:
            gold_reasonings.append(parsed_dict['gold_reasoning'])

    dataset = []
    for i, val in enumerate(questions):
        if has_gold_reason:
            dataset.append(dspy.Example(question=val, answer=answers[i], gold_reasonings=gold_reasonings[i]).with_inputs('question'))
        else:
            dataset.append(dspy.Example(question=val, answer=answers[i]).with_inputs('question'))  
        
    return dataset

hotpot_trainset = get_dataset_from_str(hotpot_qa_train_content, False)
custom_trainset = get_dataset_from_str(custom_trainset_content, False)
custom_testset = get_dataset_from_str(custom_testset_content, False)