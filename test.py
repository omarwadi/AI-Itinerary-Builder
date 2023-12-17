"""
other than these types it's considered a description and you must at least find one of these types in the user's input.

    some cases where you must not give the user description:
        if the user gives you one or more pokemon names or in general if the user gives you some names dont continue even if 
        it describes it and tell the user:
        "Please only provide me with one or two types"
        
        if you couldn't find the types or the user provided you with more than 2 types then tell the user
        "Please give me at least one type and at most two types".
        
        if you couldn't find the types or the user only provided you with One or more than Three pokemons, then tell the 
        user: 
        "Please give me at most Two pokemons or dont specify and specify at least One type and at most Two types"
        
        If the user ask you a question that is irrelevant then dont answer and say 
        "Im sorry i cant help you with that, Im a pokemon generator so Please provide me with :
            - At least one type and two types at most
        to generate for you a new Pokemon"  """

"""

    email = st.text_input(label="Enter your email", help="hey")

    if re.fullmatch(regex, email):
        print("Valid Email")

    else:
        print("Invalid Email")

"""

"""
for chunk in bot_response.split():
    full_response += chunk + " "
    time.sleep(0.05)
    # Add a blinking cursor to simulate typing
    message_placeholder.write(full_response + "▌")
message_placeholder.write(full_response)
"""
"""
message_placeholder = st.empty()
                full_response = ""
            for chunk in bot_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.write(full_response + "▌")
            message_placeholder.write(full_response)

"""


# openai.api_key = "18f15fe7944c47fdaf38614a94403a90"
# openai.api_base = "https://pokemongpt.openai.azure.com/"
# openai.api_type = "azure"n
# openai.api_version = "2023-05-15"

"""
    This function takes nothing and it adds the response schema with the description template
    so the model outputs the response in a way similir to what we want
"""
# name = ResponseSchema(name="Place Name", description="Whats the name of the place?")

# location = ResponseSchema(name="Location", description="Where the place is?")
# price = ResponseSchema(
#     name="Price",
#     description="How much does the place cost to go to? (Expensive, Cheap, Free) and how much exactly?",
# )

# response_schemas = [
#     name,
#     location,
#     price,
# ]

# output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
# format_instructions = output_parser.get_format_instructions()

"""

from email_validator import validate_email, EmailNotValidError


async def check(email: str):
   

    try:
        v = validate_email(email)
        email = v["email"]
        return True
    except EmailNotValidError as e:
        return False


"""
