# nebula_prompt_calculator.py

def nebula_prompt_calculator(n, m, k):
    """
    Calculate prompt usage for Nebula based on:
    n = prompts per share per test run
    m = number of shares in the index
    k = rebalancing period in months
    """
    
    print()
    print("Total number of prompts : ", int(n*m*k/1000), "thousand")
    print("Total tokens (assume 10k tokens per call): ", int(n*m*k*10000/1000000), "million" )



if __name__ == "__main__":
    try:
        n = int(input("Enter number of prompts per share per test run (n): "))
        m = int(input("Enter number of shares in the index (m): "))
        k = int(input("Enter rebalancing period in months (k): "))

        nebula_prompt_calculator(n, m, k)
    except ValueError:
        print("Please enter valid integer values for n, m, and k.")
