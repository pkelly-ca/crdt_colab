from states import *

# Necessary Libraries
from IPython.display import Markdown, display
import shutil


# State Function Calls
# Process:
# 1) Runtime -> Run before
# 2) Run this cell by clicking on play button to the left
#
# Output for all the states appears below
#

# Change the following parameters as needed
write_sheet = True # #Make True to enable writes of csv files to the directory tree
#Working states
states = ["AK","AL","AR","CA","CT","DC","FL","GA","GU",
          "HI","IL","IN","KY","LA","MA","MN","MS","MT"
          "NC","NE","NH","NM","NY","OR","PA","TN","TX",
          "VA","VT","WA"]

#failing states - standalone run
#states = ["DE","ID","MD","ME","MO","RI","UT","WI","WY"]

#states using drivers
#states = ["AK","MO","HI","ID","NC","NH","WI","WY"]
#states=["AL"]

# You should not need to edit below this line

maindir = 'crdt_' +  date.today().strftime('%m%d%y')
if os.path.exists(maindir):
  if os.path.exists(maindir + '_old'):
    shutil.rmtree(maindir + '_old')
  shutil.move(maindir,maindir + '_old')

for state in states:
  print("\n")
  display("***" + state + " Output:***")
  func = globals()["run" + state]
  func(None,write_sheet)
