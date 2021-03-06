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
write_sheet = False # #Make True to enable writes of csv files to the directory tree
#Working states
states = ["AK","AL","AR","CA","CT","DC","DE","FL","GA","GU",
          "HI","ID","IL","IN","KY","LA","MA","MD","ME","MI",
          "MO","MN","MS","MT","NC","ND","NE","NH","NM","NV",
          "NY","OR","PA","RI","SD","TN","TX","UT","VA","VT",
          "WA","WI","WY"]


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

failed_states_list = []
for state in states:
  print("\n")
  display("***" + state + " Output:***")
  try:
    func = globals()["run" + state]
    func(None,write_sheet)
  except Exception as e:
    display("Skipping state %s due to error: %s" % (state, str(e)))
    failed_states_list.append(state)

if len(failed_states_list) > 0:
  display("These states failed to run: %s" % ', '.join(failed_states_list))
else:
  display("ALL STATES PASSED!!!")
