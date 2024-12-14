from nltk.tree import Tree
from utils import verify
from datetime import datetime
from isoduration import parse_duration

def acrop_format_error(question, answer):
    word = question
    poem = answer
    if isinstance(poem, str):
        initial_word = ''.join([x.strip()[0] for x in poem.split('\n') if len(x.strip())>0])
    elif isinstance(poem, list):
        initial_word = ''.join([x.strip()[0] for x in poem])
    else:
        raise TypeError()
    if word.upper() == initial_word.upper():
        return 1
    return -1

def conll_format_error(question, answer):
    src = ' '.join(question)
    sent = answer.strip()
    tags = [
        ('<PER>', '</PER>'),
        ('<ORG>', '</ORG>'),
        ('<LOC>', '</LOC>'),
        ('<MISC>', '</MISC>'),
    ]
    for htag, ttag in tags:
        if sent.count(htag) != sent.count(ttag):
            return -1
        sent = sent.replace(htag, '').replace(ttag, '')
    if sent.replace(' ', '') != src.replace(' ', ''):
        return -1
    return 1

def mtt_format_error(question, answer):
    src = question['src']
    term = question['term']
    term = eval(term)
    for src_term, tgt_term in term.items():
        if (src_term in src) and (tgt_term not in answer):
            return -1
    return 1

def ptb_format_error(question, answer):
    CLAUSE_LABELS = ['S', 'SBAR', 'SBARQ', 'SINV', 'SQ']
    PHRASE_LABELS = ['ADJP', 'ADVP', 'CONJP', 'FRAG', 'INTJ', 'LST', 'NAC', 'NP', 'NX', 'PP', 'PRN', 'PRT', 'QP', 'RRC', 'UCP', 'VP', 'WHADJP', 'WHADVP', 'WHNP', 'WHPP', 'X']
    WORD_LABELS = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', '.', ',', '#', '$', ':', '-LRB-', '-RRB-', '``', "''", '-NONE-']
    ALL_LABELS = CLAUSE_LABELS + PHRASE_LABELS + WORD_LABELS
    res = answer
    sentence = question.strip()
    try:
        tree = Tree.fromstring(res.strip())
    except:
        return -1
    errors = []

    if ''.join(tree.leaves()).replace(' ', '') != sentence.replace(' ', ''):
        return -1

    for pos in tree.treepositions(order='preorder'):
        subtree = tree[pos]
        if isinstance(subtree, str):  # skip
            continue
        label = subtree.label()
        if label not in ALL_LABELS:
            return -1
        if len(subtree) == 0:
            return -1
        if isinstance(subtree[0], str):  # leaf
            return -1
        else:
            if (label not in CLAUSE_LABELS) and (label not in PHRASE_LABELS):
                return -1

    return 1

def squad_format_error(question, answer):
    context = question['context'].lower()
    if answer.lower() not in context:
        return -1
    return 1

def subtitle_format_error(question, answer):
    sentence = question
    sentence = sentence.strip()
    res = answer.strip()
    def all_chars(sent):
        return sent.replace('<eol>', '').replace('<eob>', '').replace(' ', '')
    if all_chars(sentence) != all_chars(res):
        return -1
    blocks = res.split('<eob>')
    blocks = [x.split('<eol>') for x in blocks]
    for i, block in enumerate(blocks):
        if len(block) > 2:
            return -1
        for j, line in enumerate(block):
            line = line.strip()
            if len(line) > 42:
                return -1
    return 1

def ftime_format_error(question, answer):
    src = question
    tgt = answer
    tag=src["tag"]
    if not tgt:
        return -1
    if tag == 3:
        # print(tgt)
        try:
            data = tgt.split("/")
            repeat = data[0] #R-1
            tgt = data[1] # YYYYMMDDTHHMMSS
            duration = data[2] # P0Y0M0DT0H0M0S
        except Exception as e:
            return -1

        if "R" not in repeat:
            return -1
        try:
            duration =  parse_duration(duration)
        except:
            return -1
    # length
    if len(tgt)!= len("YYYYMMDDTHHMMSS"):
        return -1
    else:
        if 'T' not in tgt:
            return -1
        else:
            if len(tgt.split('T')[0]) != len('YYYYMMDD'):
                return -1
            if len(tgt.split('T')[1]) != len('HHMMSS'):
                return -1
            try:
                tmp = datetime.strptime(tgt.replace("?", "0"), "%Y%m%dT%H%M%S")
            except:
                return -1
    return 1

def trec_format_error(question, answer):
    ALL_LABELS = ['ABBR:abb', 'ABBR:exp', 'ENTY:animal', 'ENTY:body', 'ENTY:color', 'ENTY:cremat', 'ENTY:currency', 'ENTY:dismed', 'ENTY:event', 'ENTY:food', 'ENTY:instru', 'ENTY:lang', 'ENTY:letter', 'ENTY:other', 'ENTY:plant', 'ENTY:product', 'ENTY:religion', 'ENTY:sport', 'ENTY:substance', 'ENTY:symbol', 'ENTY:techmeth', 'ENTY:termeq', 'ENTY:veh', 'ENTY:word', 'DESC:def', 'DESC:desc', 'DESC:manner', 'DESC:reason', 'HUM:gr', 'HUM:ind', 'HUM:title', 'HUM:desc', 'LOC:city', 'LOC:country', 'LOC:mount', 'LOC:other', 'LOC:state', 'NUM:code', 'NUM:count', 'NUM:date', 'NUM:dist', 'NUM:money', 'NUM:ord', 'NUM:other', 'NUM:period', 'NUM:perc', 'NUM:speed', 'NUM:temp', 'NUM:volsize', 'NUM:weight']
    if answer.strip() not in ALL_LABELS:
        return -1
    return 1

def xdl_format_error(question, answer):
    try:
        errors = verify.verify_xdl(answer)
    except:
        return -1
    errors = [x['errors'][0] for x in errors]
    if len(errors) == 0:
        return 1
    return -1

def format_reward(task, question, answer):
    """
    This function checks the format of the answer against predefined rules for each task.

    Parameters:
    task (str): The name of the task. It should be one of the following:
        'acrop', 'conll', 'mtt', 'ptb', 'squad', 'subtitle', 'ftime', 'trec', 'xdl-generation'
    question (str, list, or dict): The question related to the answer.
    answer (str): The answer to be evaluated.

    Returns:
    int: The reward value. It returns 1 if the answer is correct, and -1 otherwise.
    """
    assert task in ['acrop', 'conll', 'mtt', 'ptb', 'squad', 'subtitle', 'ftime', 'trec', 'xdl-generation']
    if task in ['trec', 'squad', 'conll', 'subtitle', 'mtt', 'ptb', 'ftime']:
        answer = answer.split('\n\n')[0]
        answer = answer.split('\n')[0]
    elif task == 'acrop':
        raise NotImplementedError()
    elif task == 'xdl-generation':
        if '</XDL>' not in answer:
            return -1
        answer = answer.split('</XDL>')[0] + '</XDL>'
    else:
        raise NotImplementedError()

    if task == 'acrop':
        return acrop_format_error(question, answer)
    elif task == 'conll':
        return conll_format_error(question, answer)
    elif task == 'mtt':
        return mtt_format_error(question, answer)
    elif task == 'ptb':
        return ptb_format_error(question, answer)
    elif task == 'squad':
        return squad_format_error(question, answer)
    elif task == 'subtitle':
        return subtitle_format_error(question, answer)
    elif task == 'ftime':
        return ftime_format_error(question, answer)
    elif task == 'trec':
        return trec_format_error(question, answer)
    elif task == 'xdl-generation':
        return xdl_format_error(question, answer)
    else:
        raise NotImplementedError()
