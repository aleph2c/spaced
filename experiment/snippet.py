t = ['','', "a, b, c"]

def format_args(arguments):
  index = 1
  for arg in arguments:
    string1 =  "				:param%s %s: " % (index, arg)
    string1 += "				:type%s %s: " % (index, arg)
    yield (string1)
    index += 1

arguments = t[2].split(',')
arguments = [argument.strip() for argument in arguments if argument]

if len(arguments):
  fargs = format_args(arguments)
  tags = [next(fargs) for index in range(len(arguments))] 
  print(tags)

