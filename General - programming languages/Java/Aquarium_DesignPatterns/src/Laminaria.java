import java.awt.Color;
import java.awt.Graphics;

/**
 * class that build a new laminaria
 * @author oReL
 *
 */
public class Laminaria extends Immobile {

	private Color col;
	private int x;
	private int y;
	private int size;
	private AquaPanel a;
	/**
	 * c'tor
	 * @param name
	 * @param x
	 * @param y
	 * @param size
	 * @param c
	 */
	public Laminaria(String name,int x,int y,int size,Color c)
	{
		super(name);
		this.col=c;
		this.x=x;
		this.y=y;
		this.size=size;
		scs=new SeaCreatureState("Laminaria", getInfo());
	}
	/**
	 * function that is made to draw the laminaria
	 * @param Graphics g
	 */
	public void draw(Graphics g) 
	{
		g.setColor(col);
		g.fillArc(x-size/20, y-size, size/10, size*4/5, 0, 360);
		g.fillArc(x-size*3/20, y-size*13/15, size/10, size*2/3, 0, 360);
		g.fillArc(x+size/20, y-size*13/15, size/10, size*2/3, 0, 360);
		g.drawLine(x, y, x, y-size/5);
		g.drawLine(x, y, x-size/10, y-size/5);
		g.drawLine(x, y, x+size/10, y-size/5);
	}
	@Override
	/**
	 * implementation of SeaCreature
	 * @see SeaCreature
	 */
	public void drawCreature(Graphics g) 
	{
		this.draw(g);
		a.repaint();
	}
	/**
	 * function that do callback from aquapanel
	 * @param AquaPanel a to do callback
	 */
	public void setAquaPanel(AquaPanel a) 
	{
		this.a=a;
	}
	@Override
	/**
	 * function that return the info of the immobile via object[]
	 * @return Object[] info
	 */
	public Object[] getInfo()
	{
		Object[] info=new Object[4];
		info[0]=col;
		info[1]=size;
		info[2]=x;
		info[3]=y;
		return info;
	}
	@Override
	/**
	 * function that restore the info of the immobile via object[] info
	 * @param Object[] info
	 */
	public void setInfo(Object[] info)
	{
		size=(int) info[1];
		x=(int) info[2];
		y=(int) info[3];
	}
	@Override
	/**
	 * function that return the size of the Laminaria
	 * @return int size
	 */
	public int getSize() 
	{
		return size;
	}
}
