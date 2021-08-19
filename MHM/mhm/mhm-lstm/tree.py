# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 12:12:02 2019

@author: DrLC
"""

class ParsingTreeNode(object):
    
    def __init__(self, _type=None, _value=None):
        
        assert _type is not None, "None _type argument."
        assert _value is not None, "None _value argument."
        assert _type in ["T", "N"], "Invalid _type argument."
        
        self.value = _value
        self.children = []
        if _type == "T":
            self.ter = True
            self.non = False
        else:
            self.ter = False
            self.non = True
        self.parent = None
        
    def addChild(self, _node=None):
        
        assert _node is not None, "None _node argument"
        assert isinstance(_node, ParsingTreeNode), "Invalid type of _node argument"
        
        self.children.append(_node)
        _node.parent = self
        
    def __printNode(self, _depth=0, _delim="  "):
        
        for i in range(_depth):
            print (_delim, end="")
        if self.ter:
            print ("[#] <"+self.value+">")
        if self.non:
            print ("[*] ["+self.value+"]")
            
    def printSubTree(self, _depth=0, _delim="  "):
        
        self.__printNode(_depth=_depth, _delim=_delim)
        for c in self.children:
            c.printSubTree(_depth=_depth+1, _delim=_delim)
            
    def getSeq(self, _seq="", _delim=" "):
        
        if self.ter:
            if _seq == "":
                _seq = self.value
            else:
                _seq = _seq + " " + self.value
        else:
            for c in self.children:
                _seq = c.getSeq(_seq=_seq, _delim=_delim)
        return _seq
        
    def getTokens(self, _tokens=[]):
        
        if self.ter:
            _tokens.append(self.value)
        else:
            for c in self.children:
                _seq = c.getTokens(_tokens)
        return _tokens
                
    def toDict(self):
        
        _dict = {}
        _dict["value"] = self.value
        _dict["ter"] = self.ter
        _dict["non"] = self.non
        _dict["children"] = []
        for c in self.children:
            _dict["children"].append(c.toDict())
        return _dict
            
def dict2PTNode(_dict={}):
    
    if _dict['ter']:
        _node = ParsingTreeNode(_type="T", _value=_dict['value'])
    elif _dict['non']:
        _node = ParsingTreeNode(_type="N", _value=_dict['value'])
    else:
        assert False
    for _c_dict in _dict['children']:
        _c_node = dict2PTNode(_c_dict)
        _node.addChild(_c_node)
    return _node
        
def ParsingTree(_lines=[], _delim="\t"):
    
    depth = 0
    root = None
    for idx in range(len(_lines)):
        _line = _lines[idx]
        line = _line.strip()
        _depth = depth
        depth = 0
        for ch in _line:
            if not ch == _delim:
                break
            depth += 1
        if len(line) <= 0:
            continue
        if line[:3] == "[*]":
            _type = "N"
        elif line[:3] == "[#]":
            _type = "T"
        else:
            return None
        _value = line[5:-1]
        _node = ParsingTreeNode(_type=_type, _value=_value)
        if depth == 0:
            root = _node
            curr = root
        else:
            while _depth >= depth:
                curr = curr.parent
                _depth -= 1
            curr.addChild(_node=_node)
            curr = _node
    return root
    
if __name__ == "__main__":
    
    with open("helloworld.tree", "r") as f:
        lines = f.readlines()
    root = ParsingTree(_lines=lines, _delim="\t")
    root.printSubTree()
    print (root.getSeq())