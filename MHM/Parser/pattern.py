# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 13:50:49 2020

@author: DrLC & Fuzy
"""

import re
import pickle, gzip
INDENT = "  "

def extractStr(tokens):
    mask2token, token2mask = {}, {}
    result = []
    cnt = 0 
    for token in tokens:
        if "'" in token or '"' in token: 
            if token2mask.get(token) == None:
                mask = "<_str%d_>"%cnt
                token2mask[token] = mask
                mask2token[mask] = token
                cnt += 1
            result.append(token2mask[token])
        else:
            result.append(token)
    return result, token2mask, mask2token

def recoverStr(tokens, mask2token):
    result = []
    for token in tokens:
        if token.startswith("<_str"):
            result.append(mask2token[token])
        else:
            result.append(token)
    return result

def _go4next(tokens, token, curIdx):
    n = len(tokens)
    while curIdx < n and tokens[curIdx] != token:
        curIdx += 1
    if curIdx == n:
        return -1
    else:
        return curIdx 

def _go4match(tokens, startToken, curIdx):
    endToken = ""
    if startToken == "(":
        endToken = ")"
    elif startToken == "[":
        endToken = "]"
    elif startToken == "{":
        endToken = "}"
    else:
        assert False

    indent = 0
    n = len(tokens)
    while curIdx < n:
        if tokens[curIdx] == startToken:
            indent += 1
        elif tokens[curIdx] == endToken:
            indent -= 1
            if indent == 0:
                break
        curIdx += 1
    if curIdx == n:
        return -1
    else:
        return curIdx

def _tokens2stmts(tokens, level=0):
    le_paren = 0
    idx = 0
    n = len(tokens)
    res = ""
    res += INDENT * level
    inAssign = False
    while idx < n:
        t = tokens[idx]
        res += t + " "
        if t == "(":
            le_paren += 1
        elif t == ")":
            le_paren -= 1
            if le_paren == 0:               # in case of "if ((a=b)!=0)"
                inAssign = False            
        elif t == ";" and le_paren == 0:    # in case of ";" in "for (int i=0; i<10: i++)"
            res += "\n"
            if idx != n - 1:
                res += INDENT * level
            inAssign = False
        elif t in [";", ",", ":", "?"]:
            inAssign = False
        elif t == "{" and not inAssign:     # in case of "int a [ 10 ] [ 10 ] = { { 0 }, { 0 } };"
            startIdx = idx + 1
            endIdx = _go4match(tokens, "{", idx)
            res += "\n"
            res += _tokens2stmts(tokens[startIdx: endIdx], level + 1)
            res += "\n"
            if endIdx+1 != n:
                res += INDENT * level
            idx = endIdx
        elif t == "{" and inAssign:
            idx += 1
            while idx < n:
                res += tokens[idx] + " "
                if tokens[idx] == ";":
                    res += "\n"
                    if idx != n - 1:
                        res += INDENT * level
                    inAssign = False
                    break
                idx += 1
        elif t in ["=", "enum"]:
            inAssign = True;
        idx += 1
    return res

def _getIndent(str_):
    res = ""
    for ch in str_:
        if ch in ["\t", " "]:
            res += ch
        else:
            break
    return res



# return: (stmts, StmtInsPos)
#   stmts: a list of statements, where variable declaration can be inserted following
#   StmtInsPos: a list of indices, indicating which statments can be seen as statement insertion positions
def tokens2stmts(tokens):

    tokens, token2mask, mask2token = extractStr(tokens)
    tokens = _tokens2stmts(tokens)
    stmts = tokens.split("\n")
    stmts = ["" if stmt.strip() == "" else stmt for stmt in stmts]

    # Add temporary omitted "}" with corresponding indentation
    statStack = []
    newStmts = []
    for stmt in stmts:
        if stmt != "":
            newStmts.append(stmt)
            if stmt.rstrip().endswith("{"):
                statStack.append(stmt)
        elif len(statStack) != 0:
            matchStmt = statStack.pop()
            newStmts.append(_getIndent(matchStmt)+"}")
        else:
            pass
            #newStmts.append("}")

    '''
    struct a {              struct a {
        int i;                  int i;
        double j;   ==>         double j;
    }                       } c, d, e;
    c, d, e;
    '''
    pattern_uid = "[A-Za-z_][A-Za-z0-9_]*(\[[0-9]*\])*"
    pattern = "^({},\s*)*{};".format(pattern_uid, pattern_uid)
    stmts = []
    blockStack = []  
    endStruct = False # including struct & union
    for stmt in newStmts:
        #print([stmt])
        if stmt.rstrip().endswith("{"):
            stmts.append(stmt)
            blockStack.append(stmt)
        elif stmt.strip() == "}":
            stmts.append(stmt)
            if blockStack[-1].lstrip().startswith("struct") or\
              blockStack[-1].lstrip().startswith("union") or\
              blockStack[-1].lstrip().startswith("typedef"):
                endStruct = True
            blockStack.pop()
        elif endStruct:
            if stmt.strip() == ";" or re.match(pattern, stmt.replace(" ","")):
                stmts[-1] = stmts[-1] + " " + stmt
            else: 
                stmts.append(stmt)
            endStruct = False
        else:
            stmts.append(stmt)
    
    paren_n = 0
    StmtInsPos = []
    structStack = []
    for i, stmt in enumerate(stmts):
        if stmt.rstrip().endswith("{"):
            paren_n += 1
            if stmt.lstrip().startswith("struct") or stmt.lstrip().startswith("union"):
                structStack.append((stmt, paren_n))
        elif stmt.lstrip().startswith("}"):
            if structStack != [] and paren_n == structStack[-1][1]:
                structStack.pop()
            paren_n -= 1
        if structStack == []:
            StmtInsPos.append(i)

    indents = [_getIndent(stmt) for stmt in stmts]
    stmts = [recoverStr(stmt.strip().split(), mask2token) for stmt in stmts]   

    return stmts, StmtInsPos, indents

def StmtInsPos(tokens, strict=True):
    '''
    Find all possible positions to insert any statements
    '''
    statements, StmtInsPos, _ = tokens2stmts(tokens)
    res = []
    cnt, indent = 0, 0
    if not strict:
        res.append(-1)
    for i, stmt in enumerate(statements):
        cnt += len(stmt)
        if stmt[-1] == "}":
            indent -= 1
        elif stmt[-1] == "{" and stmt[0] not in ["struct", "union", "enum", "typedef"]:
            indent += 1
        if i in StmtInsPos:
            if not strict:
                res.append(cnt-1)
            elif stmt[-1]!="}" and\
                indent!=0 and\
                stmt[0] not in ['else', 'if'] and\
                not (stmt[0]=='for' and 'if' in stmt):
                res.append(cnt-1)
    return res

def DeclInsPos(tokens):
    '''
    Find all possible positions to insert variable declarations
    '''
    statements, _, _ = tokens2stmts(tokens)
    res = []
    cnt = 0
    res.append(-1)
    for stmt in statements:
        cnt += len(stmt)
        res.append(cnt-1)
    return res

def BrchInsPos(tokens):
    '''
    Find all possible positions to insert false branch that control flow will never reach
    '''
    return StmtInsPos(tokens)

def LoopInsPos(tokens):
    '''
    Find all possible positions to insert loop that has no effect
    '''
    return StmtInsPos(tokens)

def FuncInsPos(tokens):
    '''
    Find all possible positions to insert functions
    '''
    return StmtInsPos(tokens)

def _InsVis(tokens, pos):
    statements, _, indents = tokens2stmts(tokens)
    lens = [len(line) for line in statements]

    for pidx in pos:
        if pidx == -1:
            statements[0] = ["[____]"] + statements[0]
            continue
        cnt = 0
        for i, n in enumerate(lens):
            cnt += n
            if cnt > pidx:
                statements[i].append("[____]")
                break

    for indent, line in zip(indents, statements):
        print(indent, end="")
        print(" ".join(line))



# return [pos1, pos2, ...]
def InsAddCandidates(insertDict, maxLen=None):

    res = []
    for pos in insertDict.keys():
        if pos == "count":
            continue
        if maxLen is None:
            res.append(pos)
        elif int(pos) < maxLen:
            res.append(pos)
    return res

# only able to insert into legal poses, and can't insert same inserted content into same pos (return False)
def InsAdd(insertDict, pos, insertedTokenList):

    suc = True
    assert insertDict.get(pos) is not None  # this position could be inserted
    if insertedTokenList in insertDict[pos]:    # can't insert same statement
        suc = False
    else:
        insertDict[pos].append(insertedTokenList)

    if suc:
        if insertDict.get("count") is not None:
            insertDict["count"] += 1
        else:
            insertDict["count"] = 1
    return suc

# return [(pos1, 0), (pos1, 1), (pos2, 0), ...]
def InsDeleteCandidates(insertDict):

    res = []
    for key in insertDict.keys():
        if key == "count":
            continue
        if insertDict[key] != []:
            for i, _ in enumerate(insertDict[key]):
                res.append((key, i))
    return res

# what is passed in must is legal (pos, listIdx)
def InsDelete(insertDict, pos, listIdx=0):

    assert insertDict.get(pos) is not None
    assert insertDict[pos] != []
    if len(insertDict[pos]) <= listIdx:
        assert False
    del insertDict[pos][listIdx]
    assert insertDict.get("count") is not None
    insertDict["count"] -= 1

# return complete tokenlist by inserting corresponding token list (insertions)
def InsResult(tokens, insertDict):

    result = []
    if insertDict.get(-1) is not None:
        for tokenList in insertDict[-1]:
            result += tokenList
    for i, t in enumerate(tokens):
        result.append(t)
        if insertDict.get(i) is not None:   # so it's a legal insertion position
            for tokenList in insertDict[i]:
                result += tokenList
    return result



# Usage: for each token in <tokens>, find "end token" that indicating where the corresponding statement ends.
#   tokens: token list
#   optimize: optimize level
#       0 => Label index of statement end token within single line.                         E.g. "for(...) sum++;" => index of ";"
#       1 => Label index of recurrent statement block end token (including "switch").       E.g. "while() {}"、"do {} while ();"、"for() {}"
#       2 => Label index of if-else statement block end token.                              E.g. "if() {} else if() {} else xxx;" => ";"
#       3 => Label index of other open statement "block" end token.                         E.g. "int() {}" => "}"       "enum { 1, 5, a }" => "}"
# Return: <list<int>> of index of end tokens
def getStmtEnd(tokens, optimize=3):

    statements, _, indents = tokens2stmts(tokens)
    heads = [stmt[0] for stmt in statements]
    ends = [stmt[-1] for stmt in statements]
    lens = [len(stmt) for stmt in statements]
    n = len(ends)
    endIndices = []

    # end token index for each line (single line statement)
    totalCnt = 0
    for i, (head, end, cnt, indent) in enumerate(zip(heads, ends, lens, indents)):
        totalCnt += cnt
        if end == ";":
            if i == n-1:
                endIndices.append(totalCnt - 1)
            elif head not in ["for", "while", "do", "if", "else", "switch"]:
                endIndices.append(totalCnt - 1)
            elif head in ["for", "while", "switch", "do"]:
                endIndices.append(totalCnt - 1)
            elif heads[i+1] != "else":
                endIndices.append(totalCnt - 1)
            else:
                endIndices.append(None)
        elif end == "}":
            if i == n-1:
                endIndices.append(totalCnt - 1)
            elif len(indents[i+1]) < len(indent):
                endIndices.append(totalCnt - 1)
            elif heads[i+1] != "else" and heads[i+1] != "while":
                endIndices.append(totalCnt - 1)
            else:
                endIndices.append(None)
        else:
            endIndices.append(None)
    if optimize <= 0:
        return endIndices

    # end token index for each line ("for { }" & "switch { }" & "while { }" & "do { } while ();" block statement)
    totalCnt = 0
    for i, (head, end, cnt, indent) in enumerate(zip(heads, ends, lens, indents)):
        totalCnt += cnt
        if endIndices[i] != None:
            continue
        if head not in ["for", "while", "switch"]:
            continue
        if end == "{":
            curIdx = i + 1
            while curIdx < n and len(indents[curIdx]) > len(indent):
                curIdx += 1
            assert curIdx < n and heads[curIdx] == "}", "[%d]%s"%(curIdx, heads[curIdx])
            endIndices[i] = endIndices[curIdx]
    for i, (head, end, cnt, indent) in enumerate(zip(heads, ends, lens, indents)):
        totalCnt += cnt
        if endIndices[i] != None:
            continue
        if head not in ["do"]:
            continue
        if end == "{":
            curIdx = i + 1
            while curIdx < n and len(indents[curIdx]) > len(indent):
                curIdx += 1
            assert curIdx < n and heads[curIdx] == "}", "[%d]%s"%(curIdx, heads[curIdx])
            curIdx += 1
            while curIdx < n and not (indents[curIdx]==indent and heads[curIdx]=="while" and ends[curIdx]==";"):
                curIdx += 1
            assert curIdx < n and heads[curIdx] == "while", "[%d]%s"%(curIdx, heads[curIdx])
            endIndices[i] = endIndices[curIdx]
    if optimize <= 1:
        return endIndices

    # end token index for each line ("if else" statement)
    totalCnt = 0
    for i, (head, end, cnt, indent) in enumerate(zip(heads, ends, lens, indents)):
        totalCnt += cnt
        if endIndices[i] != None:
            continue
        if head not in ["if", "else"]:
            continue
        curTotalCnt = totalCnt
        curIdx = i
        while True:
            curIdx += 1
            while curIdx < n and len(indents[curIdx]) > len(indent):
                curTotalCnt += lens[curIdx]
                curIdx += 1
            assert curIdx < n   # because all single if/else statements have been processed in o-0
            if endIndices[curIdx] != None:
                endIndices[i] = endIndices[curIdx]
                break
    for i, (head, end, cnt, indent) in enumerate(zip(heads, ends, lens, indents)):
        totalCnt += cnt
        if endIndices[i] != None:
            continue
        if not (head=="}" and i+1<n and heads[i+1]=="else"):
            continue
        endIndices[i] = endIndices[i+1]
    if optimize <= 2:
        return endIndices

    # end token index for each line (left "{ }" block statement, e.g. "int main() {}" & "enum { ...; }")
    # WARNING! This WILL occur to assertion error. NO GUARANTEE!
    totalCnt = 0
    for i, (head, end, cnt, indent) in enumerate(zip(heads, ends, lens, indents)):
        totalCnt += cnt
        if endIndices[i] != None:
            continue
        if end == "{":
            curIdx = i + 1
            while curIdx < n and len(indents[curIdx]) > len(indent):
                curIdx += 1
            assert curIdx < n and heads[curIdx] == "}", "[%d]%s"%(curIdx, heads[curIdx])
            endIndices[i] = endIndices[curIdx]
    
    res = []
    for cnt, endIdx in zip(lens, endIndices):
        res += [endIdx] * cnt
    return res

def IfElseReplacePos(tokens, endPoses):
    pos = []
    n = len(tokens)
    for i, t in enumerate(tokens):
        if t == "if":     # only "if {} else {}", ont process "else if {}"
            ifIdx = i
            conditionEndIdx = _go4match(tokens, "(", ifIdx)
            if tokens[conditionEndIdx + 1] == "{":
                ifBlockEndIdx = _go4match(tokens, "{", conditionEndIdx + 1)
            else:
                ifBlockEndIdx = _go4next(tokens, ";", conditionEndIdx + 1)
            if not (ifBlockEndIdx + 1 < n and tokens[ifBlockEndIdx + 1] == "else"):
                continue
            if tokens[ifBlockEndIdx + 2] == "if":   # in case of "else if {}"
                continue
            elseBlockEndIdx = endPoses[ifBlockEndIdx + 1]
            pos.append([ifIdx, conditionEndIdx, ifBlockEndIdx, elseBlockEndIdx])
    return pos

def IfElseReplace(tokens, pos):
    beforeIf = tokens[:pos[0]]
    codition = tokens[pos[0]+2:pos[1]]
    if tokens[pos[1]+1] == "{":
        blockIf = tokens[pos[1]+2:pos[2]]
    else:
        blockIf = tokens[pos[1]+1:pos[2]+1]
    if tokens[pos[2]+2] == "{":
        blockElse = tokens[pos[2]+3:pos[3]]
    else:
        blockElse = tokens[pos[2]+2:pos[3]+1]
    afterElse = tokens[pos[3]+1:]
    res = beforeIf + ["if", "(", "!", "("] + codition + [")", ")", "{"] + blockElse + ["}", "else", "{"] + blockIf + ["}"] + afterElse
    return res 

def IfReplacePos(tokens, endPoses):
    pos = []
    n = len(tokens)
    for i, t in enumerate(tokens):
        if t == "if":
            ifIdx = i
            conditionEndIdx = _go4match(tokens, "(", ifIdx)
            if tokens[conditionEndIdx + 1] == "{":
                ifBlockEndIdx = _go4match(tokens, "{", conditionEndIdx + 1)
            else:
                ifBlockEndIdx = _go4next(tokens, ";", conditionEndIdx + 1)
            if ifBlockEndIdx + 1 < n and tokens[ifBlockEndIdx + 1] == "else":   # in case of "if {} else {}", only process "if {} xxx"
                continue
            pos.append([ifIdx, conditionEndIdx, ifBlockEndIdx])
    return pos

def IfReplace(tokens, pos):
    beforeIf = tokens[:pos[0]]
    condition = tokens[pos[0]+2:pos[1]]
    if tokens[pos[1]+1] == "{":
        body = tokens[pos[1]+2:pos[2]]
    else:
        body = tokens[pos[1]+1:pos[2]+1]
    afterIf = tokens[pos[2]+1:]
    # if (a) {} => if (!a); else {}
    res = beforeIf + ["if", "(", "!", "("] + condition + [")", ")", ";", "else", "{"] + body + ["}"] + afterIf
    return res 

def For2WhileReplacePos(tokens, endPoses):
    pos = []
    for i, t in enumerate(tokens):
        if t == "for":
            forIdx = i
            conditionEndIdx = _go4match(tokens, "(", forIdx)
            if tokens[conditionEndIdx + 1] == "{":
                blockForEndIdx = _go4match(tokens, "{", conditionEndIdx)
            else:
                blockForEndIdx = endPoses[conditionEndIdx + 1]
            condition1EndIdx = _go4next(tokens, ";", forIdx)
            condition2EndIdx = _go4next(tokens, ";", condition1EndIdx + 1)
            pos.append([forIdx, condition1EndIdx, condition2EndIdx, conditionEndIdx, blockForEndIdx])
    return pos

def For2WhileRepalce(tokens, pos):
    beforeFor = tokens[:pos[0]]
    condition1 = tokens[pos[0]+2:pos[1]+1]
    condition2 = tokens[pos[1]+1:pos[2]]
    condition3 = tokens[pos[2]+1:pos[3]] + [";"]
    if tokens[pos[3]+1] == "{":
        body = tokens[pos[3]+2:pos[4]]
    else:
        body = tokens[pos[3]+1:pos[4]+1]
    afterFor = tokens[pos[4]+1:]
    if beforeFor != [] and beforeFor[-1] in [";", "{", "}"]:
        res = beforeFor + condition1 + ["while", "("] + condition2 + [")", "{"] + body + condition3 + ["}"] + afterFor
    else:
        res = beforeFor + ["{"] + condition1 + ["while", "("] + condition2 + [")", "{"] + body + condition3 + ["}", "}"] + afterFor
    return res    

def While2ForReplacePos(tokens, endPoses):
    pos = []
    n = len(tokens)
    for i, t in enumerate(tokens):
        if t == "while":
            whileIdx = i
            conditionEndIdx = _go4match(tokens, "(", whileIdx)
            if conditionEndIdx + 1 < n and tokens[conditionEndIdx + 1] == ";":   # in case of "do {} while ();"
                continue
            if tokens[conditionEndIdx + 1] == "{":
                blockWhileEndIdx = _go4match(tokens, "{", conditionEndIdx)
            else:
                blockWhileEndIdx = endPoses[conditionEndIdx + 1]
            pos.append([whileIdx, conditionEndIdx, blockWhileEndIdx])
    return pos

def While2ForRepalce(tokens, pos):
    beforeWhile = tokens[:pos[0]]
    condition = tokens[pos[0]+2:pos[1]]
    if tokens[pos[1]+1] == "{":
        body = tokens[pos[1]+2:pos[2]]
    else:
        body = tokens[pos[1]+1:pos[2]+1]
    afterWhile = tokens[pos[2]+1:]
    res = beforeWhile + ["for", "(", ";"] + condition + [";", ")", "{"] + body + ["}"] + afterWhile
    return res 



if __name__ == "__main__":
    
    with gzip.open('./data/oj.pkl.gz', "rb") as f:
        d = pickle.load(f)
        raw = d['raw_tr'] + d['raw_te']

    '''for i, code in enumerate(raw):
        if "struct" in code:
            print("%d-----------------------------------------"%i)
            pos = StmtInsPos(code) #pos = StmtInsPos(code)
            _InsVis(code, pos)
            print("%d~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"%i)
            stmts, _, indents = tokens2stmts(code)
            endTokenIndices = getStmtEnd(code, 3)
            for stmt, endIdx, indent in zip(stmts, endTokenIndices, indents):
                if endIdx != None:
                    print("{:30}".format(" ".join(code[endIdx-3:endIdx+1])), end="")
                else:
                    print("{:30}".format(" "), end="")
                print(indent + " ".join(stmt))'''

    import random


    # TEST INSERT & DELETE
    import pycparser
    parser = pycparser.c_parser.CParser()
    inserts = [
        ";",
        "if ( false ) ;",
        "if ( true ) ;",
        "if ( false ) ; else ;"]
    inserts = [insert.split(" ") for insert in inserts]
    for _ in range(1000):
        code = random.sample(raw, 1)[0]
        poses = StmtInsPos(code, strict=True) # True to make sure that parse succeed (insert to position within block)
        insertDict = dict([(pos, []) for pos in poses])
        _InsVis(code, poses)
        print("-----------------------------------")
        suc_cnt = 0
        for _ in range(10):
            candis = InsAddCandidates(insertDict)
            insIdx = random.randint(0, len(candis)-1)
            pos = candis[insIdx]
            instIdx = random.randint(0, len(inserts)-1)
            inst = inserts[instIdx]
            if InsAdd(insertDict, pos, inst):
                suc_cnt += 1
            _InsVis(InsResult(code, insertDict), [])
            parser.parse(" ".join(InsResult(code, insertDict)))
            print("------------- INSERT ---------------", insertDict["count"])
        for _ in range(suc_cnt): 
            candis = InsDeleteCandidates(insertDict)
            delIdx = random.randint(0, len(candis)-1)
            pos, listIdx = candis[delIdx]
            InsDelete(insertDict, pos, listIdx)
            _InsVis(InsResult(code, insertDict), [])
            parser.parse(" ".join(InsResult(code, insertDict)))
            print("------------- REMOVE --------------", insertDict["count"])
        print()

    # TEST FOR => WHILE
    '''code = raw[2333]
    _InsVis(code, [])
    endPoses = getStmtEnd(code, 3)
    poses = For2WhileReplacePos(code, endPoses)
    for pos in poses:
        res = For2WhileRepalce(code, pos)
        print("-----------------------------------")
        _InsVis(res, [])'''

    # TEST WHILE => FOR
    '''code = raw[233]
    _InsVis(code, [])
    endPoses = getStmtEnd(code, 3)
    poses = While2ForReplacePos(code, endPoses)
    for pos in poses:
        res = While2ForRepalce(code, pos)
        print("-----------------------------------")
        _InsVis(res, [])'''

    # TEST FINAL IF ELSE
    '''code = raw[233]
    _InsVis(code, [])
    endPoses = getStmtEnd(code, 3)
    poses = IfElseReplacePos(code, endPoses)
    for pos in poses:
        res = IfElseReplace(code, pos)
        print("-----------------------------------")
        _InsVis(res, [])'''

    # TEST SINGLE IF
    '''code = raw[233]
    _InsVis(code, [])
    endPoses = getStmtEnd(code, 3)
    poses = IfReplacePos(code, endPoses)
    for pos in poses:
        res = IfReplace(code, pos)
        print("-----------------------------------")
        _InsVis(res, [])'''

