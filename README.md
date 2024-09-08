
## Pikachu-RL
<div align="center">
    <img src="https://github.com/user-attachments/assets/14596822-0bc1-4c62-8442-9f7994de092a" width="300">
    <p>
        The initial thought dump: <a href="https://docs.google.com/document/d/1I2xRX6pEAe9g4CLbFlDP9znAXdIukIu3dyK-sVX_UBU/edit">The Pikachu Project</a>
    </p>
</div>

While the high level premise for this project is very simple(though a moonshot for me, probably): Build a robotic pikachu, what exactly do I mean by it and the path to do so has changed multiple times.  
Here's what the super high level goal currently is:  
```
    Build a quadruped that looks like the pokemon, can receive external stimulus (audio, video) and can respond to this stimulus through motion (walk, turn, play around) and facial expressions (a screen showing it's face).
```  
### The components
This repo contains the part of the brain responsible for the motor control.  
The policy trained here would receive a medium level stimulus (direction, speed to walk in), and would generate the required motor angles to achieve that motion. A raspi probably has enough compute required to run a small policy network that will do this in real time.  

The thinking part of the brain is in [Pikachu-Brain](https://github.com/anirudhs001/Pikachu-Brain). This will take in the high level stimulus (the audio and video signal) and generate the facial expressions to be put on the screen and medium level control to be sent to the Pikachu-RL policy. Since this will be probably be done via beefy LLMs, this code would run on a beefier system and would communicate with the other modules via TCP sockets (see PikaClient.py and PikaServer.py in [Pikachu-Brain](https://github.com/anirudhs001/Pikachu-Brain))

Work on the peripheral nervous system is yet to begin. This would convert the actions received from the RL policy into signals to be sent to the servos. This will be run on the raspi as well.  

### How this works
I have modded the [ant](https://gymnasium.farama.org/environments/mujoco/ant/) env already existing in mujoco - changed the orientation of the legs, and moved the leg joint to where real quadrupeds(dogs, cats etc) have knees, and added some more sensors.  

<table width="100%">
  <tr align="center" width="100%">
    <td style="text-align: center; width="100%">
      <img src="https://github.com/user-attachments/assets/b23b1349-c28a-4465-9367-5f3145c77070" alt="Description of image 1" style="width: 250px; height: auto;"/>
      <p>Mujoco's original ant</p>
    </td>
    <td style="text-align: center; width="100%">
      <img src="https://github.com/user-attachments/assets/e91f71d4-ced3-46e8-93af-45b89ca6693f" alt="Description of image 2" style="width: 250px; height: auto;"/>
      <p>The modded ant that I use</p>
    </td>
  </tr>
</table>

So to make this work:
<ul>
    <li>replace ant.xml in mujoco/assets (setting up a symlink works too)</li>
    <li>install mujoco, gymnasium etc</li>
</ul>

#### [Notion](https://www.notion.so/RL-d7cacda151e94646ac156a7e04403478?pvs=4)
I have tried to record some of my observations here. check it out, there are some cool videos and graphs there.

#### What are these folders in this repo:
These are some of the different algorithms I have tried so far. I have tried both implementing some of them myself, and using already written implementations from stablebaselines.  
Apart from this, the Env folder contains the Environment classes which defines the mujoco env in which our agent lives. There are a couple of variations in the Env folder, and some of the algorithm folders might have their own Envs themselves. The Env class provide the observations, the step function to move in the mujoco env. The custom logic for the reward is defined in this class too.

#### Todos
<ul>
    <li> probably add a readme in each folder </li>
</ul>
