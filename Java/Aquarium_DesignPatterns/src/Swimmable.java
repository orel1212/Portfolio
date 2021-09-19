import java.awt.Color;
import java.awt.Graphics;
import java.util.concurrent.CyclicBarrier;

/**
 * class Fish which represent a swimmable in the aquarium
 * @author oReL
 *
 */
public abstract class Swimmable extends Thread implements SeaCreature,Cloneable,MarineAnimal
{
	protected int horSpeed;
	protected int verSpeed;
	protected SeaCreatureState scs;
	public Swimmable() 
	{
		horSpeed = 0;
		verSpeed = 0;
	}
	/**
	 * c'tor that build a swimmable with horspeed and verspeed
	 * @param horspeed,verspeed
	 */
	public Swimmable(int hor, int ver) 
	{
		horSpeed = hor;
		verSpeed = ver;
	}
	/**
	 * returns the SeaCreatureState of the swimmable
	 * @return SeaCreatureState of the swimmable
	 */
	public SeaCreatureState getSeaCreatureState()
	{
		return scs;
	}
	/**
	 * function that returning the horspeed of the swimmable
	 * @return int horSpeed
	 */
	public int getHorSpeed() { return horSpeed; }
	/**
	 * function that returning the verspeed of the swimmable
	 * @return int verSpeed
	 */
	public int getVerSpeed() { return verSpeed; }
	/**
	 * function that set new value in the horspeed of the swimmable
	 * @param int hor which is the new horspeed
	 */
	public void setHorSpeed(int hor) { horSpeed = hor; }
	/**
	 * function that set new value in the verspeed of the swimmable
	 * @param int ver which is the new verspeed
	 */
	public void setVerSpeed(int ver) { verSpeed = ver; }
	/**
	 * getAnimalName that return the name of the Swimmable
	 * @return String name
	 */
	abstract public String getAnimalName();
	/**
	 * function that is made to draw the swimmable
	 * @param Graphics g
	 */
	abstract public void drawAnimal(Graphics g);
	/**
	 * function that suspend the swimmable from moving
	 */
	abstract public void setSuspend();
	/**
	 * function  that made to do notify and to wakeup the swimmable
	 */
	abstract  public void setResume();
	/**
	 * function that made to start the cyclickbarrier process,using callback to do the cyclickbarrier
	 * @param CyclicBarrier b
	 */
	abstract public void setBarrier(CyclicBarrier b);
	/**
	 * function that return the size of the swimmable
	 * @return int size
	 */
	abstract public int getSize();
	/**
	 * function that increase the eats of the swimmable by 1
	 */
	abstract public void eatInc();
	/**
	 * function that return the eats of the swimmable 
	 * @return int numofeats
	 */
	abstract public int getEatCount();
	/**
	 * function that return color of the swimmable
	 * @return Color col
	 */
	abstract public Color getColor();
	/**
	 * function that made to stop the cyclicbarrier
	 */
	abstract public void StopBarrier();
	/**
	 * function that do callback from aquapanel
	 * @param AquaPanel a to do callback
	 */
	abstract public void setAquaPanel(AquaPanel a);
	/**
	 * function that made to update the attributes after duplicate
	 * @param hspeed,vspeed,color and size
	 */
	public abstract void UpdateDuplicateAttributes(int hspeed,int vspeed,Color c,int size);
	/**
	 * implementation of Cloneable
	 * @see Cloneable interface
	 */
	public abstract Swimmable clone();
	/**
	 * function that check if the state is Hungry
	 * @return true if yes,false if satiated
	 */
	public abstract boolean isHungry();
	
	public abstract void SetColor(Color c);
	/**
	 * function that return the info of the swimmable via object[]
	 * @return Object[] info
	 */
	public abstract Object[] getInfo();
	/**
	 * function that restore the info of the swimmable via object[] info
	 * @param Object[] info
	 */
	public abstract void setInfo(Object[] info);
	/**
	 * function that make new memento to save
	 * @see Memento class
	 * @return Memento obj
	 */
	public Memento MakeMomento()
	{
		return new Memento(scs);
	}
	/**
	 * function setState that set the state to the swimmable
	 * @param hs
	 */
	public abstract void setState(HungerState hs);
}
