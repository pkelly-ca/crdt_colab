# Top Level Directory
# Authors: Pat Kelly and Carol Brandmaier-Monahan
from states import *

# Necessary Libraries
from IPython.display import Markdown, display
import shutil
import getopt,sys
import os
import datetime
from datetime import date, timedelta

# notify Slack in case of errors                                                                                                                               
from slack import WebClient
from slack.errors import SlackApiError

begin_run = time.time()


write_sheet = False  #Default to not write to S3
write_local = False
debug = False
states = []
states_all = ["AK","AL","AR","CA","CO","CT","DC","DE","FL","GA","GU",
          "HI","ID","IL","IN","KY","LA","MA","MD","ME","MI",
          "MO","MN","MS","MT","NC","ND","NE","NH","NM","NV",
          "NY","OR","PA","RI","SD","TN","TX","UT","VA","VT",
          "WA","WI","WY"]

args_list  = sys.argv[1:]

# Options
opts = "hawls:"
long_opts = ["help", "all", "state", "write", "local"]
try:
    # Parsing argument
    args, vals = getopt.getopt(args_list, opts, long_opts)

    # checking each argument
    for arg, val in args:
        if arg in ("-h", "--help"):
            print ("Use:\n-a OR --all to run all states;\n-s <ST> OR --state <ST> to run a single state;\n-w OR --write to write to S3.")
            sys.exit()

        elif arg in ("-a", "--all"):
            print ("Running All")
            states = states_all

        elif arg in ("-s", "--state"):
            print (("Running State = (% s)") % (val))
            states = [str(val)]
            debug = True

        elif arg in ("-w", "--write"):
            print ("Writing Output")
            write_sheet = True # #Make True to enable writes of csv files to the directory tree

        elif arg in ("-l", "--local"):
            print ("Writing Locally")
            write_local = True # Overrides S3 write, writes to local database (Pat's google drive for now)


except getopt.error as err:
    # output error, and return with an error code
    print (str(err))


# You should not need to edit below this line

if len(states) == 0:
    sys.exit("ERROR: Must supply either -a or -s argumemnt")

#maindir = 'crdt_' +  date.today().strftime('%m%d%y')
#if os.path.exists(maindir):
#  if os.path.exists(maindir + '_old'):
#    shutil.rmtree(maindir + '_old')
#  shutil.move(maindir,maindir + '_old')


failed_states_list = []
for state in states:
  print("\n")
  display("***" + state + " Output:***")
  start = time.time()
  if debug == False:
    try:
      func = globals()["run" + state]
      func(write_local,write_sheet)
    except Exception as e:
      display("Skipping state %s due to error: %s" % (state, str(e)))
      failed_states_list.append(state)
  else:
    try:
      func = globals()["run" + state]
      func(write_local,write_sheet)
      display("STATE PASSED")
    except Exception as e:
      display("STATE FAILED")
      display(e)
  end = time.time()
  duration = end - start
  print('%s run time = %.2f s' % (state, duration))

if len(failed_states_list) > 0:
  message = "CRDT auto data fetcher states failed to run: %s" % ', '.join(failed_states_list)
  display(message)
  # notify slack                                                                                                                                               
  slack_api_token = os.environ.get('SLACK_API_TOKEN')
  slack_channel = os.environ.get('SLACK_CHANNEL')
  if slack_api_token and slack_channel:
    client = WebClient(token=slack_api_token)
    try:
      response = client.chat_postMessage(
        channel=slack_channel,
        text=message)
    except SlackApiError as e:
      display("Could not notify Slack, received error")
  else:
    display('SLACK_API_TOKEN and/or SLACK_CHANNEL environment variable not set')
elif debug == False:
  display("ALL STATES PASSED!!!")

end_run = time.time()
time_run = end_run - begin_run
print('Total run time = %.2f s' % time_run )

