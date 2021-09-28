# HF_Turtlebot_Base
HF domain for Curriculum Transfer



To run the curriculum example, create a virtual environment in virtualenv/conda/etc based off of the requirements.txt, then:

```
$ python test_curr.py --mode <mode>
```

Where `<mode>` is `simpledqn` or `dqnlambda` depending on your preference.

The RL algorithm is in the SimpleDQN.py file. 
The curriculum is manually generated, and the parameters for each task are given in the `test_curr.py` file. 
