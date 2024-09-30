# Example model
Example model - the dynare bkk.mod - wrapped and made to work

This obviously has the same compute location, making it much simpler, but it demonstrates how one might wrap a dynare model and make it callable from Python, while retaining lineage.

There are obviously many bits missing, including the lineage components. For now, it does the following:
1. Demo wrapping and calling a dynare model from the python script
2. Demo getting the data in and out, and the parameters being passed
3. Test how difficult this is and how much boilerplate / modification is required.

All done side-of-desk.

## Structuring
This is written as a python package to make it pip installable in the script (notebook) which is being used to orchestrate

