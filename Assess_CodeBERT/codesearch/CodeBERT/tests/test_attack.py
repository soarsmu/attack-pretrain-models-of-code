import sys 
sys.path.append("..") 
sys.path.append(".") 

from attack import get_masked_code_by_position

def test_get_masked_code_by_position():
    tokens = ["Hello", ",", "I", "am", "a", "boy", "."]
    posistions = [0, 2, 3]

    masked_token_list = get_masked_code_by_position(tokens, posistions)
    assert len(masked_token_list) == len(posistions)
