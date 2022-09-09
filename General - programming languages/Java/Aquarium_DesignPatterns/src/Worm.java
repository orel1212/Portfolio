import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Polygon;

/**
 * class that represents the worm as the food of the animals in the aquarium
 * @author oReL
 *
 */
public class Worm 
{
	private Color color;
	private static Worm worm=null;
	/**
	 * set the color of the worm to color
	 * @param color
	 */
	public void SetColor(Color color)
	{
		this.color=color;
	}
	/**
	 * static function to make the singleton
	 * @param color
	 * @return new worm if not created,or if already created so the existed one
	 */
	public static Worm CreateWorm(Color color)
	{
		if(worm==null)
		{
			worm=new Worm(color);
		}
		else
		{
			worm.SetColor(color);
		}
		return worm;
	}
	/**
	 * private c'tor to avoid creating new one from outside
	 * @param color
	 */
	private Worm(Color color)
	{
		this.color=color;
	}
	/**
	 * function that draw the worm at the aquarium
	 * @param g
	 * @param width of the aquapanel
	 * @param height of the aquapanel
	 */
	public void drawWorm(Graphics g,int width,int height) 
	{
		Graphics2D g2 = (Graphics2D) g;
	    g2.setStroke(new BasicStroke(3));
	    g2.setColor(Color.red);
	    g2.drawArc(width/2, height/2-5, 10, 10, 30, 210);	   
	    g2.drawArc(width/2, height/2+5, 10, 10, 180, 270);
	    g2.setStroke(new BasicStroke(1));
	}
		
}
