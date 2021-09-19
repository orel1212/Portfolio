
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;

/**
 * class that build a new Zostera
 * @author oReL
 *
 */
public class Zostera extends Immobile {

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
	public Zostera(String name,int x,int y,int size,Color c)
	{
		super(name);
		this.col=c;
		this.x=x;
		this.y=y;
		this.size=size;
		scs=new SeaCreatureState("Zostera", getInfo());
	}
	/**
	 * function that is made to draw the Zostera
	 * @param Graphics g
	 */
	public void draw(Graphics g) 
	{
		Graphics2D g2=(Graphics2D)g;
		g2.setStroke(new BasicStroke(3));
		g2.setColor(col);
		g.drawLine(x, y, x, y-size);
		g.drawLine(x-2, y, x-10, y-size*9/10);
		g.drawLine(x+2, y, x+10, y-size*9/10);
		g.drawLine(x-4, y, x-20, y-size*4/5);
		g.drawLine(x+4, y, x+20, y-size*4/5);
		g.drawLine(x-6, y, x-30, y-size*7/10);
		g.drawLine(x+6, y, x+30, y-size*7/10);
		g.drawLine(x-8, y, x-40, y-size*4/7);
		g.drawLine(x+8, y, x+40, y-size*4/7);
		g2.setStroke(new BasicStroke(1));
	}
	@Override
	/**
	 * implementation of SeaCreature
	 * @see SeaCreature
	 */
	public void drawCreature(Graphics g) 
	{
		this.draw(g);
		
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
	 * function that return the size of the Zostera
	 * @return int size
	 */
	public int getSize() 
	{
		return size;
	}

}