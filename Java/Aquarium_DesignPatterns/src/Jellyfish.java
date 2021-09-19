import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.util.ArrayList;
import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;

/**
 * class Fish which represent a jellyfish in the aquarium
 * @author oReL
 *
 */
public class Jellyfish extends Swimmable
{
	private Color col;
	private int x_dir;
	private int x_front;
	private int y_front;
	private int x_flag;
	private int y_flag;
	private int size;
	private int numofeats;
	private double new_hspeed;
	private double new_vspeed;
	private boolean wait;
	private CyclicBarrier cb;
	private boolean barrierstate;
	private AquaPanel a;
	private int frequency;
	private int timestobehungry;
	private HungerState hs;
	private ArrayList<Observer> observers;
	/**
	 * c'tor that build a new jellyfish
	 * @param horspeed,verspeed ,color,x_Dir,x,y and size to build the jellyfish
	 */
	public Jellyfish(int horspeed,int verspeed,Color c,int x ,int y,int size,int frequency)
	{
		super(horspeed,verspeed);
		col=c;
		this.x_front=x;
		this.y_front=y;
		this.size=size;
		numofeats=0;
		wait=false;
		barrierstate=false;
		x_flag=y_flag=1;
		this.frequency=frequency;
		hs=new Hungry();
		hs.doAction(this);
		timestobehungry=frequency;
		observers=new ArrayList<Observer>(1);
		scs=new SeaCreatureState("Jellyfish", getInfo());
	}
	/**
	 * c'tor that build a new Jellyfish
	 * @param horspeed,verspeed ,color,x_Dir,x,y, size and frequency,x_flag,y_Flag to build the fish
	 */
	public Jellyfish(int horspeed,int verspeed,Color c,int x_dir,int x ,int y,int size,int frequency,int x_flag,int y_flag)
	{
		super(horspeed,verspeed);
		col=c;
		this.x_dir=x_dir;
		this.x_front=x;
		this.y_front=y;
		this.size=size;
		numofeats=0;
		wait=false;
		barrierstate=false;
		this.x_flag=x_flag;
		this.y_flag=y_flag;
		this.frequency=frequency;
		hs=new Satiated();
		hs.doAction(this);
		timestobehungry=frequency;
		observers=new ArrayList<Observer>(1);
		scs=new SeaCreatureState("Jellyfish", getInfo());
	}
	@Override
	/**
	 * getAnimalName which is override from Swimmable,return the name of the jellyfish
	 * @return String name
	 */
	public String getAnimalName()
	{
		return "Jellyfish";
	}
	
	@Override
	/**
	 * function that is made to draw the jellyfish
	 * @param Graphics g
	 */
	public void drawAnimal(Graphics g) 
	{
		int numLegs;
		if(size<40)
		numLegs = 5;
		else if(size<80)
		numLegs = 9;
		else
		numLegs = 12;
		g.setColor(col);
		g.fillArc(x_front - size/2, y_front - size/4, size, size/2, 0, 180);
		for(int i=0; i<numLegs; i++)
		g.drawLine(x_front - size/2 + size/numLegs + size*i/(numLegs+1),
		y_front, x_front - size/2 + size/numLegs + size*i/(numLegs+1),
		y_front+size/3);
	}
	/**
	 * function that do callback from aquapanel
	 * @param AquaPanel a to do callback
	 */
	public void setAquaPanel(AquaPanel a) 
	{
		this.a=a;
		observers.add(a);
	}
	@Override
	/**
	 * function that suspend the jellyfish from moving
	 */
	public synchronized void setSuspend() 
	{
		wait=true;
		
	}

	@Override
	/**
	 * function that is synchronized to do notify and to wakeup the jellyfish
	 */
	public synchronized void setResume() 
	{
		wait=false;
		notify();
		
	}

	@Override
	/**
	 * function that made to start the cyclickbarrier process,using callback to do the cyclickbarrier
	 * @param CyclicBarrier b
	 */
	public void setBarrier(CyclicBarrier b)
	{
		cb=b;
		double v_old=Math.sqrt(horSpeed*horSpeed+verSpeed*verSpeed);
		double v_new=v_old;
		double k=Math.abs( ((double)(y_front - a.getHeight()/2)) / ((double)(x_front - a.getWidth()/2)));
		new_hspeed=v_new/(Math.sqrt(k*k+1));
		new_vspeed=new_hspeed*k;
		barrierstate=true;
	}

	@Override
	/**
	 * function that return the size of the jellyfish
	 * @return int size
	 */
	public int getSize() 
	{
		return size;
	}

	@Override
	/**
	 * function that increase the eats of the jellyfish by 1
	 */
	public void eatInc() 
	{
		this.numofeats+=1;
		
	}
	/**
	 * function that made to stop the cyclicbarrier
	 */
	public void StopBarrier()
	{
		barrierstate=false;
	}
	@Override
	/**
	 * function that return the eats of the jellyfish 
	 * @return int numofeats
	 */
	public int getEatCount() 
	{
		return numofeats;
	}
	@Override
	/**
	 * function that return color of the jellyfish
	 * @return Color col
	 */
	public Color getColor() 
	{
		return col;
	}
	/**
	 * function that override the run in thread to use for the jellyfish,using callback of aquapanel to do repaint
	 */

	public void run()
	{
		while(true)
		{
			if(wait)
			{
				try {
					synchronized (this)
					{
						this.wait();
					}
					
				} catch (InterruptedException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
			else if(barrierstate)
			{
				try
				{
					cb.await();
				} 
				catch (InterruptedException e) 
				{
						e.printStackTrace();
				} 
				catch (BrokenBarrierException e) 
				{
						e.printStackTrace();
				}
				if(x_front>a.getWidth()/2)
				{
					x_flag = -1;
					x_dir=0;
				}
							
				else if(x_front<a.getWidth()/2)
				{
					x_flag=1;
					x_dir=1;
				}
				else
				{
					new_hspeed=0;//move only with the v_speed cuz he reached the right place at the x_front to eat the food
				}
				if(y_front>=a.getHeight()/2)
					y_flag=-1;
				else
					y_flag=1;
				x_front+=(int)new_hspeed*x_flag;
				y_front+=(int)new_vspeed*y_flag;
					
				if((Math.abs(x_front-a.getWidth()/2) <= 5) && (Math.abs(y_front-a.getHeight()/2) <= 5))
				{
					eatInc();
					hs=new Satiated();
					hs.doAction(this);
					timestobehungry=frequency;
					a.stopDrawFood();
					a.StopBarrier();
				}
			}
			else
				{
				if(hs.getClass()==Satiated.class)
				{
					timestobehungry--;
					if (timestobehungry==0)
					{
						hs=new Hungry();
						hs.doAction(this);
						for(int i=0;i<observers.size();i++)
							observers.get(i).update();
					}
				}
				x_front += horSpeed*x_flag;
				y_front += verSpeed*y_flag;
				
				Dimension d=a.getSize();
				if(x_front > d.width-size)
				{
					x_dir=0;
					x_flag = -1;
					x_front = d.width-size;
				}
				else if(x_front < 20)
				{
					x_flag = 1;
					x_front = 20;
			   	}

				if(y_front > d.height-20)
				{
					y_flag = -1;
					y_front = d.height-20;
				}
				else if(y_front < 20)
				{
					y_flag = 1;
					y_front = 20;
				}
			}
			try {
				sleep(50);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			
			a.repaint();
		}
	}
	@Override
	/**
	 * implementation of SeaCreature
	 * @see SeaCreature
	 */
	public void drawCreature(Graphics g)
	{
		this.drawAnimal(g);
		
	}
	/**
	 * function that made to update the attributes after duplicate
	 * @param hspeed,vspeed,color and size
	 */
	@Override
	public void UpdateDuplicateAttributes(int hspeed,int vspeed,Color c,int size) 
	{
		horSpeed=hspeed;
		verSpeed=vspeed;
		col=c;
		this.size=size;	
	}
	@Override
	/**
	 * implementation of Cloneable
	 * @see Cloneable interface
	 */
	public Swimmable clone() 
	{
		return new Jellyfish(horSpeed,verSpeed,col,x_front ,y_front,size,frequency);
	}
	/**
	 * function that check if the state is Hungry
	 * @return true if yes,false if satiated
	 */
	public boolean isHungry()
	{
		if(hs instanceof Hungry)
			return true;
		return false;
	}
	/**
	 * implementation of setColor from Swimmable
	 * @sea SetColor in swimmable
	 */
	@Override
	public void SetColor(Color c) 
	{
		this.col=c;
	}
	@Override
	public void PaintFish() 
	{
		// TODO Auto-generated method stub
		
	}
	/**
	 * function that return the info of the jellyfish via object[]
	 * @return Object[] info
	 */
	@Override
	public Object[] getInfo()
	{
		Object[] info=new Object[6];
		info[0]=col;
		info[1]=size;
		info[2]=x_front;
		info[3]=y_front;
		info[4]=horSpeed;
		info[5]=verSpeed;
		return info;
	}
	/**
	 * function that restore the info of the jellyfish via object[] info
	 * @param Object[] info
	 */
	@Override
	public void setInfo(Object[] info)
	{
		this.col=(Color)info[0];
		size=(int) info[1];
		x_front=(int) info[2];
		y_front=(int) info[3];
		horSpeed=(int) info[4];
		verSpeed=(int) info[5];
	}
	/**
	 * override of setState from swimmable
	 * @see swimmable for more info
	 */
	@Override
	public void setState(HungerState hs)
	{
		this.hs=hs;
		
	}
}
