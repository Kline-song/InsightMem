"""Deterministic whole-message process rules for assistant messages.

A message is dropped from the selectable candidates only when every clause is a
process narration with no concrete finding. Anything uncertain is retained, and
raw source text is never changed or deleted.
"""
import re

from .parse import PROTECT, TRANSITION

# --- phrase rules ---

INTRO=r"(?:(?:ok(?:ay)?|good|great|alright|right)[,.!]\s*)?(?:(?:now|next|first|then|finally),?\s*)?(?:let me|let's|i will|i'll|i need to)\s+(?:(?:also|now|first|just|quickly|carefully)\s+){0,3}"
TARGET=r"(?:`[\w./\\:-]+(?:\(\))?`|[A-Za-z_][\w./\\-]*(?:\(\))?)"
OBJECT=r"(?:(?:the|a|an|this|that)\s+)?(?:(?:relevant|related|existing|current|remaining|correct|right|source|test|actual|changed|updated|final|complete|full|entire|new|rest of the|rest of|different)\s+){0,4}(?:codebase|repository|repo|project|implementation|code|source|files?|tests?|test suite|test file|source files|code files|method|function|class|module|diff|changes?|fix|edit|result|imports?|dependencies|directory|structure|context|definition|usage|callers?|logs?)"
OBJ=rf"(?:{OBJECT}|{TARGET})(?:\s+(?:of|in|for|around)\s+(?:{OBJECT}|{TARGET}))?"
END=r'[.!:]?'
RULES_V1=[
 ('empty',re.compile(r'\s*')),
 ('ack',re.compile(r'(?:ok(?:ay)?|great|good|done|understood|sure|thanks|thank you|all done|all set|looks? good)'+END,re.I)),
 ('inspect',re.compile(INTRO+rf'(?:read|inspect|examine|review|explore|look at|check|search for|find|locate)\s+{OBJ}(?:\s+(?:first|now|again|in detail))?'+END,re.I)),
 ('inspect_goal',re.compile(INTRO+r'(?:explore|inspect|examine|review|read|look at)\s+(?:the )?(?:codebase|repository|repo|project|relevant files|code|implementation)(?: structure)?\s+to\s+(?:understand|find|locate|identify)\s+(?:the )?(?:structure|current implementation|relevant (?:code|source|files|code files|source files)|issue|problem)(?: first)?'+END,re.I)),
 ('verify',re.compile(INTRO+r'verify\s+(?:(?:the |all )?(?:final |complete )?(?:state of )?(?:the |all |my )?(?:changes?|changed files|fix|edit|diff|result|function|changed area))(?:\s+(?:looks? correct|is correct|works|by reading (?:it|the relevant section|the file)))?'+END,re.I)),
 ('review',re.compile(INTRO+r'do\s+(?:a )?(?:final |complete )?review\s+of\s+(?:(?:all )?the |all )?(?:changes|fix|diff|changed files)'+END,re.I)),
 ('run_tests',re.compile(INTRO+r'(?:run|re-run|rerun|try running|execute)\s+(?:(?:the|all|some)\s+)?(?:(?:existing|new|relevant|related|full|complete|broader|unit|regression|specific)\s+){0,3}(?:tests?|test suite)(?:\s+(?:again|now|together))?(?:\s+(?:to verify (?:the (?:fix|changes)|everything works|the fix works)|to make sure (?:everything (?:works|passes)|nothing is broken)))?'+END,re.I)),
 ('generic_test',re.compile(INTRO+r'(?:test|try)\s+again'+END,re.I)),
 ('summary',re.compile(INTRO+r'(?:provide|write|give|prepare)\s+(?:a |the )?summary(?:\s+of\s+(?:the (?:changes(?: made)?|fix)|what was (?:done|implemented)))?'+END,re.I)),
 ('complete_then_summary',re.compile(r'The fix is complete(?: and verified)?[.!]\s+'+INTRO+r'(?:provide|write|give|prepare)\s+(?:a |the )?summary(?:\s+of\s+(?:the (?:changes(?: made)?|fix)|what was (?:done|implemented)))?'+END,re.I)),
 ('tool_choice',re.compile(INTRO+r'(?:use|try)\s+(?:Bash(?: with)? grep|Bash to grep|Bash|grep|Django\x27s test runner|the test runner|pytest|tox)(?:\s+(?:for this|instead))?'+END,re.I)),
 ('generic_understand',re.compile(INTRO+r'understand\s+(?:the )?(?:issue|problem|codebase|structure)'+END,re.I)),
 ('chinese_process',re.compile(r'(?:现在|接下来|首先|然后)?(?:我来|让我|我会)?(?:查看|检查|阅读|浏览)(?:一下)?(?:相关)?(?:代码|文件|实现|测试|项目结构)[。！!：:]?')),
]
RULES_V1 += [
 ('summarize',re.compile(INTRO+r'summarize(?: what was (?:done|implemented)| the (?:changes|fix))?'+END,re.I)),
 ('structure',re.compile(INTRO+r'explore (?:the )?(?:codebase|repository|project|repo) structure(?: first)?'+END,re.I)),
 ('verify_file',re.compile(INTRO+r'verify (?:the )?(?:final state of (?:the )?files?|both changes|changes)(?: (?:looks? correct|are correct))?'+END,re.I)),
 ('verify_by_tests',re.compile(INTRO+r'verify (?:the )?(?:fix|changes) by running (?:the )?(?:relevant |existing |new )?tests'+END,re.I)),
 ('implement_generic',re.compile(INTRO+r'(?:start implementing|implement (?:the )?(?:fix|changes))(?: now)?'+END,re.I)),
 ('track_task',re.compile(INTRO+r'(?:mark (?:the )?task as complet(?:e|ed)|update (?:the )?task tracking)'+END,re.I)),
 ('runtime_navigation',re.compile(INTRO+r'(?:find (?:the )?right (?:Python/pytest|python|Python|pytest)(?: (?:interpreter|executable))?(?: to use)?|find (?:the )?right way to run tests|check how to run tests in this environment|try (?:using |with )?Django\x27s test runner instead)'+END,re.I)),
]
STATUS_PREFIX=re.compile(r'(?:The (?:fix|implementation|task) is complete(?: and verified)?)[.!]\s+',re.I)
# Negative information and concrete relationships always override generic matches.
PROTECT_V1=re.compile(r'```|\n\s*(?:[-*]|\d+[.)])\s|(?:==|!=|<=|>=|:=|=>|->)|\b(?:because|since|but|however|although|unless|except|if|whether|when|while|therefore|means|returns?|raises?|produces?|must|should|cannot|error|failed|failure|regression|None|True|False|null|zero|empty)\b|因为|但是|除非|如果|报错|失败|根因|返回|必须|不能',re.I)

def _phrase_low_information(text):
 if not isinstance(text,str):return False
 bare=text.strip()
 if bare.startswith(('[tool_use','[tool_result]')):return False
 if bare.startswith('[thinking]'):bare=bare[len('[thinking]'):].strip()
 if len(bare)>360 or PROTECT_V1.search(bare):return False
 for _, pattern in RULES_V1:
  if pattern.fullmatch(bare):return True
 prefix=STATUS_PREFIX.match(bare)
 if prefix and _phrase_low_information(bare[prefix.end():]):return True
 return False

# --- compositional grammar ---
MAX_CHARS=2000
MAX_CLAUSES=10
# A target may name files/functions; a target alone is not an observed finding.
ACTION=r'(?:look\s+(?:at|for|into)|search(?:\s+for)?|re-?read|read|inspect|examine|explore|review|check|verify|find|locate|see|get|run|re-run|rerun|execute|try|use|provide|give|write|prepare|summarize|mark|update|implement|start|modify|edit|fix|change|apply|restore|create|add)'
PREFIX=re.compile(r"^(?:(?:ok(?:ay)?|good|great|alright|right)[,!]\s*)?(?:(?:now|next|first|then|finally|also),?\s*)?(?:(?:let me|let's|i will|i'll|i need to|i want to|we need to)\s+)?(?:(?:also|now|first|just|quickly|carefully|further|again)\s+){0,4}(?="+ACTION+r'\b)',re.I)
READ=re.compile(r'^(?:look\s+(?:at|for|into)|search(?:\s+for)?|re-?read|read|inspect|examine|explore|review|check|verify|find|locate|see|get)\s+(.+)$',re.I)
TEST=re.compile(r'^(?:run|re-run|rerun|execute|try running)\s+(.+)$',re.I)
# Explicit proposition/condition signals are retained, including within a plan.
FACT=re.compile(r'\b(?:because|since|therefore|however|although|unless|except|if|whether|when|while|but|returns?|raises?|produces?|preserves?|requires?|causes?|supports?|handles?|contains?|implements?|accepts?|rejects?|omits?|allows?|prevents?|changes?\s+from|must|should|cannot|can\x27t|won\x27t|None|True|False|null|zero|empty|is|are|was|were|has|have|can|may|might|found|observed|discovered|shows?|seems?|appears?|expects?|needs?|failed|fails?|passed|passes|fixed|removed|added)\b|因为|但是|除非|如果|报错|失败|根因|返回|必须|不能',re.I)
# Purpose clauses are accepted only from a generic, bounded verification vocabulary.
GOAL=re.compile(r'(?:understand (?:the )?(?:issue|problem|code|codebase|structure|current (?:state|implementation)|(?:existing|current|full) implementation|full (?:context|construction logic)|standard pattern|available tests)|(?:find|locate|identify) (?:the )?(?:relevant (?:files|code|method|source files)|(?:right|correct) (?:method|files|code)(?: to patch)?|code to (?:change|patch))|(?:verify|confirm|check|ensure|make sure)(?: that)? (?:(?:everything|it|they|the tests) (?:works|work|passes|pass)|(?:the )?(?:fix|changes|edit)(?: (?:is|are))? (?:correct|works|work|looks correct|look correct)|nothing (?:is )?broken|the (?:fix|changes)|they pass)|see (?:(?:the )?current state|what happens|if (?:they|it|the tests) pass))',re.I)
TARGET=re.compile(r"[\w\s`./\\:()'\"\-,]+",re.UNICODE)
CODE=re.compile(r'(?<!\w)\d+(?!\w)|```|\[tool_(?:use|result)\b|(?:==|!=|<=|>=|:=|=>|->)|[{}]|\n\s*(?:[-*]|\d+[.)])\s|\b(?:def|class|assert|return|raise|import|from)\s+\w',re.I)
ACK=re.compile(r'(?:ok(?:ay)?|good|great|alright|right|understood|sure|thanks|thank you)',re.I)
GENERIC=re.compile(r"(?:(?:provide|give|write|prepare) (?:a |the )?summary(?: of (?:all )?(?:the )?(?:changes(?: made)?|fix|what was (?:done|implemented)))?|summarize(?: (?:the )?changes(?: made)?| what was (?:done|implemented))?|(?:implement|start implementing)(?: (?:the )?(?:fix|changes))?(?: now| first)?|(?:mark (?:the )?task (?:as )?completed?|update (?:the )?task (?:status|tracking|list)))",re.I)

EDIT_TARGET=r"(?:(?:the|a|an|this|that|another|second|first|next)\s+)?(?:(?:remaining|relevant|existing|current|new|proper|comprehensive|actual|changed|final|first|second|next)\s+){0,3}(?:fix(?:es)?|changes?|code|implementation|imports?|tests?|test cases|test file|tasks?|task list|location|(?:method|function|class|property|docstring|test file)\s+(?:`[\w.]+(?:\(\))?`|[A-Za-z_]\w*(?:\(\))?)|`[\w./]+(?:\(\))?`(?: (?:class|method|function|property|docstring))?)"
EDIT=re.compile(r'(?:update|modify|edit|fix|change|apply|restore|create|write|add)\s+('+EDIT_TARGET+r')(?:\s+(?:in|for)\s+(?:`[\w./]+`|[A-Z][A-Za-z0-9_]+))?(?:\s+(?:first|now))?',re.I)
INTENT=re.compile(r"^(?:(?:ok(?:ay)?|good|great|alright|right)[,!]\s*)?(?:(?:now|next|first|then|finally|also),?\s*)?(?:let me|let's|i will|i'll|i need to|i want to|we need to)\b|^(?:now|next|first|then|finally),?\s+",re.I)
def _clauses(bare):
 # Protect punctuation inside identifier quotes so foo.py / foo() remain a target.
 quotes=[]
 def mask(m):quotes.append(m.group());return 'QUOTEDTOKEN'+str(len(quotes)-1)
 masked=re.sub(r'`[^`\n]+`',mask,bare)
 masked=re.sub(r'[.!?](?=\s|$)|[;\n]+', '\n',masked)
 masked=re.sub(r'\s*(?:,?\s+and\s+(?:then\s+)?|,?\s+then\s+)(?=(?:(?:let me|i will|i\x27ll)\s+)?'+ACTION+r'\b)', '\n',masked,flags=re.I)
 out=[]
 for part in masked.split('\n'):
  part=part.strip().rstrip(':').strip()
  if not part:continue
  part=re.sub(r'QUOTEDTOKEN(\d+)',lambda m:quotes[int(m[1])],part)
  out.append(part)
 return out

def _purpose(target):
 parts=re.split(r'\s+(?:in order to|to|so that|so i can)\s+',target,maxsplit=1,flags=re.I)
 if len(parts)==2:
  if not GOAL.fullmatch(parts[1].strip()):return None
  return parts[0].strip()
 return target.strip()

def _clause(text, inherited=False):
 if ACK.fullmatch(text):return True
 if not inherited and not INTENT.search(text):return False
 m=PREFIX.match(text)
 if not m:return False
 body=text[m.end():].strip()
 if GENERIC.fullmatch(body):return True
 if EDIT.fullmatch(body) and not FACT.search(body):return True
 rm=READ.fullmatch(body)
 if rm:
  target=_purpose(rm[1])
  if not target or len(target)>300 or FACT.search(target):return False
  # Imperative navigation only; no asserted predicate, equation or list of changes.
  return bool(TARGET.fullmatch(target))
 tm=TEST.fullmatch(body)
 if tm:
  target=_purpose(tm[1])
  if not target or FACT.search(target):return False
  # No literal commands or expected outputs; a test name/file is only a target.
  return bool(re.fullmatch(r"(?:(?:the|all|a|some|another)\s+)?(?:(?:existing|new|relevant|related|full|complete|broader|unit|regression|specific|manual|proper|remaining)\s+)*(?:[\w`./:-]+\s+){0,4}(?:tests?|test suite|test suites|test file)(?: (?:again|now|together|one more time))?",target,re.I))
 return False

def _grammar_low_information(text):
 if not isinstance(text,str):return None
 bare=text.strip()
 if bare.startswith('[thinking]'):bare=bare[len('[thinking]'):].strip()
 if not bare or len(bare)>MAX_CHARS or 'QUOTEDTOKEN' in bare or CODE.search(bare):return None
 clauses=_clauses(bare)
 if not clauses or len(clauses)>MAX_CLAUSES:return None
 # AND across clauses, never any()/prefix matching: one finding preserves all.
 inherited=False
 for c in clauses:
  if not _clause(c,inherited):return None
  inherited=inherited or bool(INTENT.search(c))
 return 'assistant_process_grammar_v2'


def low_information(text):
    """True when an assistant message narrates process only."""
    if not isinstance(text, str):
        return False
    if _phrase_low_information(text) or _grammar_low_information(text):
        return True
    bare = text.removeprefix("[thinking]").strip()
    if TRANSITION.fullmatch(bare) and not PROTECT.search(bare):
        return True
    return bool(re.fullmatch(
        r"(?:The )?(?:file|source|code|module)(?: [\w./-]+)? (?:was )?(?:examined|inspected|read|reviewed|located)[.!]?",
        bare, re.I))
