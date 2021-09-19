import java.awt.Color;

/**
 * class that represent the decorator and implements MarineAnimal
 * @author oReL
 *
 */
public class MarineAnimalDecorator implements MarineAnimal
{
	Color color;
	MarineAnimal ma;
	/**
	 * c'tor
	 * @param ma
	 * @param c
	 */
	public MarineAnimalDecorator(MarineAnimal ma,Color c) 
	{
		this.ma=ma;
		this.color=c;
	}
	@Override
	/**
	 * implemenetation of MarineAnimal
	 * @see MarineAnimal
	 */
	public void PaintFish() 
	{
		if(ma instanceof Swimmable)
			((Swimmable)ma).SetColor(color);
	}

}
