# The language of document QA. Optional values include: en, zh, and None. The tokenization method for F1-score calculation differs among the three languages.
DEFAULT_LANGUAGE = "None"

# Q&A prompt for document QA – replace the {} placeholders with actual content from the dataset when used.
doc_qa_prompt_zh = [
    """
    阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{input}\n回答：
    """
]

doc_qa_prompt_en = [
    """
    Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:
    """
]

multi_answer_prompt = [
    """
    Please read the following text and answer the questions below.\n
    Text: {context}\n
    What is the correct answer to this question: {question}\n
    Choices: \n (A) {choice_A} \n (B) {choice_B} \n (C) {choice_C} \n (D) {choice_D} \n 
    Let's think step by step. Based on the above, what is the single, most likely answer choice?\n
    Format your response as follows: "The correct answer is (insert answer here)'
"""
]

match_patterns = [
    r"The correct answer is \(([A-D])\)",
    r"The correct answer is ([A-D])",
    r"The \(([A-D])\) is the correct answer",
    r"The ([A-D]) is the correct answer",
]
