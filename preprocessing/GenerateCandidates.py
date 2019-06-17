# coding: utf-8



import Levenshtein
import pickle
import re


def getallngrams(s, topsize=None):
    topsize = len(s) if topsize is None else topsize
    ngrams = set()
    i = 0
    while i < len(s):
        j = i + 1
        while j <= min(len(s), i + topsize):
            ngram = tuple(s[i:j])
            j += 1
            ngrams.add(ngram)
        i += 1
    return ngrams


id_outdegree_dict = pickle.load(open("data/id_outdegree_dict.pkl", "rb"))
name_id_dict = pickle.load(open("data/fb2m_name_id_dict.pkl", "rb"))
ent_rel_dict = pickle.load(open("data/fb2m_ent_rel.pkl", "rb"))

question = "who is the publisher of nhl 08 ?"
words = question.split(" ")


pattern = "the|a|an|of|on|at|by"
allnames = name_id_dict.keys()
with open("data/stopwords.pkl", "rb") as f:
    stopwords = pickle.load(f)

def sub_candidates(words, theta=400):
    ngrams = getallngrams(words)
    cur_dict = dict()
    known_fragments = []
    null_fragments = []
    for ngram in ngrams:
        parts = " ".join(ngram)
        if parts in stopwords:
            continue
        if name_id_dict.get(parts) is not None:
            known_fragments.append(parts)
            cur_set = name_id_dict.get(parts)
            if len(cur_set) > theta:
                cur_set = sorted(cur_set,
                                 key=lambda x: id_outdegree_dict.get(x),
                                 reverse=True)[0:theta]
                cur_set = set(cur_set)
            cur_dict[parts] = cur_set
        else:
            null_fragments.append(parts)
    known_fragments = sorted(known_fragments, key=lambda x:len(x))
    n = len(known_fragments)
    target_fragments = []
    for i in range(n):
        flag = True
        tempTuple_i = tuple(known_fragments[i].split(" "))
        for j in range(i+1, n):
            j_gram = getallngrams(known_fragments[j].split(" "))
            if tempTuple_i in j_gram:
                if not re.match(pattern, known_fragments[j]):
                    flag = False
        if flag:
            target_fragments.append(known_fragments[i])
    all_candidates = set()
    for fragment in target_fragments:
        all_candidates = all_candidates | cur_dict.get(fragment)
    if len(all_candidates) == 0:
        print("seconde ---")
        sencond_candidates = set()
        for gram in null_fragments:
            for name in allnames:
                distane = Levenshtein.distance(gram, name)
                if distane == 1:
                    temp_set = name_id_dict.get(name)
                    if len(temp_set) > theta:
                        cur_set = sorted(temp_set,
                                         key=lambda x: id_outdegree_dict.get(x),
                                         reverse=True)[0:theta]
                        temp_set = set(cur_set)
                    sencond_candidates = sencond_candidates | temp_set
                    target_fragments.append(gram)
        all_candidates = sencond_candidates
    return all_candidates, target_fragments

print("*****************************")
candidates2, grams = sub_candidates(words)
print(candidates2)
print(len(candidates2))


print(" | ".join(grams))
print("done!")


def generate_subcandidates(spath, tpath, grampath):
    # the format is : [question, golden_subid, golden_rel, golden_name]
    all_recorders = pickle.load(open(spath, 'rb'))
    n = 0
    count = 0
    writer1 = open(tpath, 'w', encoding="ascii")
    gram_writer = open(grampath, "w", encoding="utf-8")
    for line in all_recorders:
        candidates, grams = sub_candidates(line[0].split(" "))
        if line[1] in candidates:
            count += 1
        writer1.write(" ".join(candidates) + "\n")
        gram_writer.write(" | ".join(grams)+"\n")
        n += 1
        if n % 10000 == 0:
            print(" --> {}".format(n))
    print(spath)
    print("candidates: recall = {} / {} = {}".format(count, n, count * 1.0 / n))
    writer1.close()
    gram_writer.close()
    print("done!")


generate_subcandidates("train/source/train.pkl",
                       "train/subcandi_400.txt",
                       "train/grams_400.txt")

generate_subcandidates("test/source/test.pkl",
                       "test/subcandi_400.txt",
                       "test/grams_400.txt")

generate_subcandidates("valid/source/valid.pkl",
                       "valid/subcandi_400.txt",
                       "valid/grams_400.txt")
