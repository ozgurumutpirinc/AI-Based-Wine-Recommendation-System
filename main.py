import warnings
warnings.filterwarnings('ignore', category=UserWarning)
from wine_model import generate_recommendation



query = "My friend is going to Poland so what is the most popular wine and its price in Europe region?"
answer = generate_recommendation(query)
#print("\nRecommendation:\n", answer, "\n" + "-")
print("User: ", query)
print("-----------------")
print("Wine Expert: ", answer)