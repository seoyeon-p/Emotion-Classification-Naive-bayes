import re, math, collections, itertools, os
import nltk, nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist


DATA_DIR = os.path.join('emotions')
ANGER_FILE = os.path.join(DATA_DIR, 'anger.txt')
DISGUST_FILE = os.path.join(DATA_DIR, 'disgust.txt')
FEAR_FILE = os.path.join(DATA_DIR, 'fear.txt')
JOY_FILE = os.path.join(DATA_DIR, 'joy.txt')
SURPRISE_FILE = os.path.join(DATA_DIR, 'surprise.txt')



def evaluate_features(feature_select):
    angerFeatures, disgustFeatures, fearFeatures, joyFeatures, surpriseFeatures = [], [], [], [], []
    with open(ANGER_FILE, 'r',errors="ignore", encoding = "utf-8") as angerSentence:
        for i in angerSentence:
            angerWords= re.findall(r"[\w']+|[.,!?;]", i.rstrip())
            angerWords = [feature_select(angerWords), 'anger']
            angerFeatures.append(angerWords)
    with open(DISGUST_FILE, 'r',errors="ignore", encoding = "utf-8") as disgustSentence:
        for i in disgustSentence:
            disgustWords = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
            disgustWords = [feature_select(disgustWords), 'disgust']
            disgustFeatures.append(disgustWords)
    with open(FEAR_FILE, 'r',errors="ignore", encoding = "utf-8") as fearSentence:
        for i in fearSentence:
            fearWords = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
            fearWords = [feature_select(fearWords), 'fear']
            fearFeatures.append(fearWords)
    with open(JOY_FILE, 'r',errors="ignore", encoding = "utf-8") as joySentence:
        for i in joySentence:
            joyWords = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
            joyWords = [feature_select(joyWords), 'joy']
            joyFeatures.append(joyWords)
    with open(SURPRISE_FILE, 'r',errors="ignore",encoding = "utf-8") as surpriseSentence:
        for i in surpriseSentence:
            surpriseWords = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
            surpriseWords = [feature_select(surpriseWords), 'surprise']
            surpriseFeatures.append(surpriseWords)
    """
    wordFeatures = []
    with open("twitteroutput.txt","r",errors="ignore",encoding = "utf-8") as sentences:
        for i in sentences:
            label = i.split("\t")[-1].strip()
            i = i.split("\t")[0]
            words = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
            words = [feature_select(words), label]
            wordFeatures.append(words)
    # selects 3/4 of the features to be used for training and 1/4 to be used for testing
    cutOff = int(math.floor(len(words) * 3 / 4))
    trainFeatures = wordFeatures[:cutOff]
    testFeatures = wordFeatures[cutOff:]
    """

    angerCutoff = int(math.floor(len(angerFeatures) * 3 / 4))
    disgustCutoff = int(math.floor(len(disgustFeatures) * 3 / 4))
    fearCutoff = int(math.floor(len(fearFeatures) * 3 / 4))
    joyCutoff = int(math.floor(len(joyFeatures) * 3 / 4))
    surpriseCutoff = int(math.floor(len(surpriseFeatures) * 3 / 4))

    trainFeatures = angerFeatures[:angerCutoff] + disgustFeatures[:disgustCutoff] + fearFeatures[:fearCutoff] + joyFeatures[:joyCutoff] + surpriseFeatures[:surpriseCutoff]
    testFeatures = angerFeatures[angerCutoff:] + disgustFeatures[disgustCutoff:] + fearFeatures[fearCutoff:] + joyFeatures[joyCutoff:] + surpriseFeatures[surpriseCutoff:]
    classifier = NaiveBayesClassifier.train(trainFeatures)


    referenceSets = collections.defaultdict(set)
    testSets = collections.defaultdict(set)

    for i, (features, label) in enumerate(testFeatures):
        referenceSets[label].add(i)
        predicted = classifier.classify(features)
        testSets[predicted].add(i)

    print('train on %d instances, test on %d instances' % (len(trainFeatures), len(testFeatures)))
    print('accuracy:', nltk.classify.util.accuracy(classifier, testFeatures))
    #print('pos precision:', nltk.metrics.scores.precision(referenceSets['pos'], testSets['pos']))
    #print('pos recall:', nltk.metrics.scores.recall(referenceSets['pos'], testSets['pos']))
    #print('neg precision:', nltk.metrics.scores.precision(referenceSets['neg'], testSets['neg']))
    #print('neg recall:', nltk.metrics.scores.recall(referenceSets['neg'], testSets['neg']))
    classifier.show_most_informative_features(10)


def make_full_dict(words):
    return dict([(word, True) for word in words])


print('using all words as features')
evaluate_features(make_full_dict)


def create_word_scores():
    angerWords, disgustWords, fearWords, joyWords, surpriseWords = [], [], [], [], []
    with open(ANGER_FILE, 'r',errors="ignore", encoding = "utf-8") as angerSentence:
        for i in angerSentence:
            angerWord = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
            angerWords.append(angerWord)
    with open(DISGUST_FILE, 'r',errors="ignore", encoding = "utf-8") as disgustSentence:
        for i in disgustSentence:
            disgustWord = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
            disgustWords.append(disgustWord)
    with open(FEAR_FILE, 'r',errors="ignore", encoding = "utf-8") as fearSentence:
        for i in fearSentence:
            fearWord = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
            fearWords.append(fearWord)
    with open(JOY_FILE, 'r',errors="ignore", encoding = "utf-8") as joySentence:
        for i in joySentence:
            joyWord = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
            joyWords.append(joyWord)
    with open(SURPRISE_FILE, 'r',errors="ignore") as surpriseSentence:
        for i in surpriseSentence:
            surpriseWord = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
            surpriseWords.append(surpriseWord)
    angerWords = list(itertools.chain(*angerWords))
    disgustWords = list(itertools.chain(*disgustWords))
    fearWords = list(itertools.chain(*fearWords))
    joyWords = list(itertools.chain(*joyWords))
    surpriseWords = list(itertools.chain(*surpriseWords))


    word_fd = FreqDist()
    cond_word_fd = ConditionalFreqDist()
    for word in angerWords:
        word_fd[word.lower()] += 1
        cond_word_fd['anger'][word.lower()] += 1
    for word in disgustWords:
        word_fd[word.lower()] += 1
        cond_word_fd['disgust'][word.lower()] += 1
    for word in fearWords:
        word_fd[word.lower()] += 1
        cond_word_fd['fear'][word.lower()] += 1
    for word in joyWords:
        word_fd[word.lower()] += 1
        cond_word_fd['joy'][word.lower()] += 1
    for word in surpriseWords:
        word_fd[word.lower()] += 1
        cond_word_fd['surprise'][word.lower()] += 1

    anger_word_count = cond_word_fd['anger'].N()
    disgust_word_count = cond_word_fd['disgust'].N()
    fear_word_count = cond_word_fd['fear'].N()
    joy_word_count = cond_word_fd['joy'].N()
    surprise_word_count = cond_word_fd['surprise'].N()
    total_word_count = anger_word_count + disgust_word_count + fear_word_count + joy_word_count + surprise_word_count

    word_scores = {}
    for word, freq in word_fd.items():
        anger_score = BigramAssocMeasures.chi_sq(cond_word_fd['anger'][word], (freq, anger_word_count), total_word_count)
        disgust_score = BigramAssocMeasures.chi_sq(cond_word_fd['disgust'][word], (freq, disgust_word_count),total_word_count)
        fear_score = BigramAssocMeasures.chi_sq(cond_word_fd['fear'][word], (freq, fear_word_count),total_word_count)
        joy_score = BigramAssocMeasures.chi_sq(cond_word_fd['joy'][word], (freq, joy_word_count),total_word_count)
        surprise_score = BigramAssocMeasures.chi_sq(cond_word_fd['surprise'][word], (freq, surprise_word_count),total_word_count)
        word_scores[word] = anger_score + disgust_score + fear_score + joy_score + surprise_score

    return word_scores


word_scores = create_word_scores()


def find_best_words(word_scores, number):
    best_vals = sorted(word_scores.items(), key=lambda  s: s, reverse=True)[:number]
    best_words = set([w for w, s in best_vals])
    return best_words


def best_word_features(words):
    return dict([(word, True) for word in words if word in best_words])


numbers_to_test = [10, 100, 1000, 10000, 15000]
for num in numbers_to_test:
    print('evaluating best %d word features' % (num))
    best_words = find_best_words(word_scores, num)
    evaluate_features(best_word_features)