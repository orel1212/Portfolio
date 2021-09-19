import java.awt.Color;

/**
 * class that implements AbstractSeaFactory, to create a new animal
 * @author oReL
 *
 */
public class AnimalFactory implements AbstractSeaFactory 
{
	private SeaCreature sc;
	private int horspeed;
	private int verspeed;
	private Color c;
	private int x ;
	private int y;
	private int size;
	private int frequency;
	/**
	 * c'tor
	 * @param horspeed
	 * @param verspeed
	 * @param c
	 * @param x
	 * @param y
	 * @param size
	 * @param freq
	 */
	public AnimalFactory(int horspeed,int verspeed,Color c,int x ,int y,int size,int freq)
	{
		this.horspeed=horspeed;
		this.verspeed=verspeed;
		this.c=c;
		this.x=x;
		this.y=y;
		this.size=size;
		frequency=freq;
	}
	@Override
	/**
	 * produceSeaCreature implmenetation
	 * @see AbstractSeaFactory
	 */
	public SeaCreature produceSeaCreature(String type) 
	{
		if(type.equals("Fish"))
		{
			if (x==0)
				sc=new Fish(horspeed,verspeed,c,1,x,y,size,frequency);
			else
				sc=new Fish(horspeed,verspeed,c,0,x,y,size,frequency);
		}
		else
		{
			sc=new Jellyfish(horspeed,verspeed,c,x,y,size,frequency);
		}
		return sc;
	}

}
