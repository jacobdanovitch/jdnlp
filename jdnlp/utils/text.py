import re

# https://gist.github.com/carlsmith/b2e6ba538ca6f58689b4c18f46fef11c
def multi_replace(string, substitutions):
    if not substitutions:
        return string
    
    substrings = sorted(substitutions, key=len, reverse=True)
    regex = re.compile('|'.join(map(re.escape, substrings)))
    
    return regex.sub(lambda match: substitutions[match.group(0)], string)