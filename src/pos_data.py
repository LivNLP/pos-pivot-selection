import numpy as np
import re

def collect(domain):
    input_file = open("../data/gweb-%s-dev.conll"%domain,'r')
    for line in input_file:
        p = filter(None,re.split("\t",line))
        print p
    pass



if __name__ == "__main__":
    domain = "answers"
    collect(domain)