import dspy
PROMPT_SWITCH = True

if __name__ == '__main__':

    #  定义并设置大模型
    model_name = 'llama3'
    lm = dspy.OllamaLocal(model=model_name)
    dspy.settings.configure(lm=lm)

    #  定义输入输出参数 inline定义方式
    print(f"## 创建并设置大模型 {model_name} ##")
    print(f"\n## inline方式 定义输入输出参数 - start ##\n")

    question = "what is the color of the sky?"          
    summarize = dspy.ChainOfThought('question -> answer')          
    response = summarize(question=question)

    print(f"问题：{question} \n答案：{response.answer}")
    print("prompt: ")
    lm.inspect_history(n=1) if PROMPT_SWITCH else print("if you want see prompt, set PROMPT_SWITCH = True")
    print(f"\n## inline方式 定义输入输出参数 - end ##\n")

    #  定义输入输出参数 类定义方式
    print(f"\n## 类方式 定义输入输出参数 - start ##\n")

    class QA(dspy.Signature):
        """answer the question of user"""
        question = dspy.InputField()
        answer = dspy.OutputField()

    question = "what is the color of the sea?"
    summarize = dspy.ChainOfThought(QA)
    response = summarize(question=question)

    print(f"问题：{question} \n答案：{response.answer}")
    print("prompt: ")
    lm.inspect_history(n=1) if PROMPT_SWITCH else print("if you want see prompt, set PROMPT_SWITCH = True")
    print(f"\n## 类方式 定义输入输出参数 - end ##\n")
