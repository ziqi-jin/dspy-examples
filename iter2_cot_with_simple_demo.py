import dspy
PROMPT_SWITCH = True

if __name__ == '__main__':

    #  定义并设置大模型
    model_name = 'llama3'
    lm = dspy.OllamaLocal(model=model_name)
    dspy.settings.configure(lm=lm) 

    print(f"## 创建并设置大模型 {model_name} ##")
    print(f"\n## 为模板增加示例 - start ##\n")

    question = "what is the color of sky at night?"          
    #  示例内容          
    example = dspy.Example(question="what is the color of sky?", answer="the color of sky is blue, even at night")          
    summarize = dspy.ChainOfThought('question -> answer')          
    response = summarize(question=question, demos=[example])  

    print(f"问题：{question} \n答案：{response.answer}")
    print("prompt: ")
    lm.inspect_history(n=1) if PROMPT_SWITCH else print("if you want see prompt, set PROMPT_SWITCH = True")
    print(f"\n## 为模板增加示例 - end ##\n")