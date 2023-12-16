from email_validator import validate_email, EmailNotValidError


async def check(email: str):
    """a function to check if the email provided by the user is valid or not.

    Args:
        email (str): the email provided by the user.

    Returns:
        bool: a boolean value that indicates if the email is valid or not.
    """

    try:
        v = validate_email(email)
        email = v["email"]
        return True
    except EmailNotValidError as e:
        return False
