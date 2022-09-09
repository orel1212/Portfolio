
import java.awt.Color;

/**
 * class that implements AbstractSeaFactory, to create a new IMMOBILE
 * @author oReL
 *
 */
public class PlantFactory implements AbstractSeaFactory 
{
	SeaCreature sc;
	Color c;
	int x ;
	int y;
	int size;
	/**
	 * c'tor
	 * @param c
	 * @param x
	 * @param y
	 * @param size
	 */
	public PlantFactory(Color c,int x ,int y,int size)
	{
		this.c=c;
		this.x=x;
		this.y=y;
		this.size=size;
	}
	@Override
	/**
	 * produceSeaCreature implmenetation
	 * @see AbstractSeaFactory
	 */
	public SeaCreature produceSeaCreature(String type) 
	{
		if(type.equals("Zostera"))
		{
			sc=new Zostera("Zostera",x,y,size,c);
		}
		else
		{
			sc=new Laminaria("Laminaria",x,y,size,c);
		}
		return sc;
	}

}
