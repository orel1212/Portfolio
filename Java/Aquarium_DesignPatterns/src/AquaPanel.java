import java.awt.BasicStroke;
import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.GridLayout;
import java.awt.Image;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.lang.management.GarbageCollectorMXBean;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;
import java.util.concurrent.CyclicBarrier;

import javax.imageio.ImageIO;
import javax.security.auth.Subject;
import javax.swing.*;

/**
 * aquapanel class that represents the panel of the aquarium
 * @author oReL
 *
 */

public class AquaPanel extends JPanel implements ActionListener,Observer
{
	private int numofSwimmables;
	private int numofImmobiles;
	private JPanel buttonpanel;
	private JButton addanimal;
	private JButton duplicateanimal;
	private JButton addplant;
	private JButton sleep;
	private JButton wakeup;
	private JButton reset;
	private JButton food;
	private JButton info;
	private JButton exit;
	private JButton decorator;
	private ArrayList<Swimmable> Sar;
	private ArrayList<Immobile> Par;
	private JScrollPane js;
	private AquaFrame af;
	private AquaPanel ap;
	private static boolean drawfood;
	private boolean oddclicks;
	private BufferedImage image;
	private JPanel duplicatepanel;
	private DuplicateAnimalDialog aad;
	/**
	 * c'tor that build the button panel and the hash set of the aquarium
	 */
	public AquaPanel(AquaFrame af)
	{
		super();
		this.af=af;
		ap=this;
		image=null;
		this.setSize(700, 700);
		this.setLayout(new BorderLayout());
		buttonpanel=new JPanel();
		buttonpanel.setLayout(new GridLayout(1,9));
		addanimal=new JButton("Add Animal");
		duplicateanimal=new JButton("Duplicate Animal");
		addplant=new JButton("Add Plant");
		sleep=new JButton("Sleep");
		wakeup=new JButton("Wake up");
		reset=new JButton("Reset");
		food=new JButton("Food");
		info=new JButton("Info");
		exit=new JButton("Exit");
		decorator=new JButton("Decorator");
		addanimal.addActionListener(this);
		decorator.addActionListener(this);
		duplicateanimal.addActionListener(this);
		addplant.addActionListener(this);
		sleep.addActionListener(this);
		wakeup.addActionListener(this);
		reset.addActionListener(this);
		food.addActionListener(this);
		info.addActionListener(this);
		exit.addActionListener(this);
		buttonpanel.add(addanimal);
		buttonpanel.add(addplant);
		buttonpanel.add(duplicateanimal);
		buttonpanel.add(decorator);
		buttonpanel.add(sleep);
		buttonpanel.add(wakeup);
		buttonpanel.add(reset);
		buttonpanel.add(food);
		buttonpanel.add(info);
		buttonpanel.add(exit);
		this.add(buttonpanel,BorderLayout.SOUTH);
		oddclicks=true;
		numofSwimmables=0;
		Par=new ArrayList<Immobile>(5);
		Sar=new ArrayList<Swimmable>(5);
		js=null;
		drawfood=false;
	}
	/**
	 * function that set the color of the background as blue or default
	 * @param colorname which is the name of the color,either default for default or blue for blue
	 */
	public void setBackgroundColor(String colorname)
	{
		if(image!=null)
		{
			image=null;
		}
		if(colorname=="Blue")
		{
			this.setBackground(Color.blue);
		}
		else//default color
		{
			this.setBackground(UIManager.getColor ( "Panel.background" ));
		}
		repaint();
		
	}
	public static void stopDrawFood()
	{
		drawfood=false;
	}
	/**
	 * function that set the image of the background as choosed by the user
	 * @param path which is the path of the image
	 */
	public void setBackgroundImage(String path)
	{
		try {
			image = ImageIO.read(new File(path));
			this.setBackground(UIManager.getColor ( "Panel.background" ));
			repaint();
		} 
    	catch (IOException e) 
		{
			e.printStackTrace();
		}
		
	}
	/**
	 * actionperformed implementation of actionlistener interface for the fuctions of the buttons in buttonpanel
	 */
	@Override
	public void actionPerformed(ActionEvent e) 
	{
		if(e.getSource()==addanimal)
		{
			if(numofSwimmables<5)
			{
				AddAnimalDialog aad=new AddAnimalDialog(this);
			}
			else
			{
				JOptionPane.showMessageDialog(null, "There are already 5 swimmable objects at the aquarium!");
			}
			
		}
		else if(e.getSource()==decorator)
		{
			JPanelDecorator jpd=new JPanelDecorator(Sar,af,this);
			af.remove(this);
			this.setVisible(false);
			af.add(jpd);
			af.setVisible(true);
			
			
		}
		else if(e.getSource()==duplicateanimal)
		{
			if(numofSwimmables<5)
			{
				duplicatepanel=new JPanel();
				JPanel choosepanel=new JPanel();
				choosepanel.setLayout(new FlowLayout());
				JButton Submit=new JButton("Submit");
				JLabel jl=new JLabel("Enter the index:");
				JTextField indexfield=new JTextField("",20);
				Submit.addActionListener(new ActionListener() {
					
					@Override
					public void actionPerformed(ActionEvent e) 
					{
						if(AquaPanel.isInteger(indexfield.getText()))
						{
							int index=Integer.parseInt(indexfield.getText());
							if(index>=Sar.size() || index<0)
							{
								JOptionPane.showMessageDialog(null ,"You insert index that not in the range!","Error",JOptionPane.ERROR_MESSAGE);
								indexfield.setText("");
							}
							else
							{
								duplicatepanel.setVisible(false);
								af.remove(duplicatepanel);
								af.add(ap);
								ap.setVisible(true);
								aad=new DuplicateAnimalDialog(ap,index);
							}
						}
						else
						{
							JOptionPane.showMessageDialog(null,"You insert some string that not represent int!", "Error",JOptionPane.ERROR_MESSAGE);
							indexfield.setText("");
						}
						
					}
				});
				
				String [] columns={"Index","Animal","Color","Size","Hor. speed","Ver. speed"};
				String[][] rows=new String[Sar.size()][6];
				for(int i=0;i<Sar.size();i++)
				{
					rows[i][0]=""+i;
					rows[i][1]=Sar.get(i).getAnimalName();
					Color c=Sar.get(i).getColor();
					if(c==Color.blue)
					{
						rows[i][2]="Blue";
					}
					else if(c==Color.red)
					{
						rows[i][2]="Red";
					}
					else if(c==Color.yellow)
					{
						rows[i][2]="Yellow";
					}
					else if(c==Color.magenta)
					{
						rows[i][2]="Magenta";
					}
					else if(c==Color.orange)
					{
						rows[i][2]="Orange";
					}
					else if(c==Color.black)
					{
						rows[i][2]="Black";
					}
					else
						rows[i][2]="("+c.getRed()+","+c.getGreen()+","+c.getBlue()+")";
					rows[i][3]=""+Sar.get(i).getSize();
					rows[i][4]=""+Sar.get(i).getHorSpeed();
					rows[i][5]=""+Sar.get(i).getVerSpeed();
					
				}
				JTable jt=new JTable(rows,columns);
				JScrollPane jsp=new JScrollPane(jt);
				duplicatepanel.add(jsp);
				this.setVisible(false);
				af.remove(this);
				duplicatepanel.setVisible(false);
				jsp.setVisible(true);
				repaint();
				choosepanel.add(jl);
				choosepanel.add(indexfield);
				choosepanel.add(Submit);
				duplicatepanel.add(choosepanel,BorderLayout.SOUTH);
				duplicatepanel.setVisible(true);
				af.add(duplicatepanel);
				
			}
			else
			{
				JOptionPane.showMessageDialog(null, "There are already 5 swimmable objects at the aquarium!");
			}
			
		}
		else if(e.getSource()==addplant)
		{
			if(numofImmobiles<5)
			{
				AddPlantDialog aad=new AddPlantDialog(this);
			}
			else
			{
				JOptionPane.showMessageDialog(null, "There are already 5 immobile objects at the aquarium!");
			}
			
		}
		else if(e.getSource()==exit)
		{
			System.exit(0);
		}
		else if(e.getSource()==sleep)
		{
			for(int i=0;i<Sar.size();i++)
			{
				Sar.get(i).setSuspend();
			}
		}
		else if(e.getSource()==wakeup)
		{

			for(int i=0;i<Sar.size();i++)
			{
				Sar.get(i).setResume();
				
			}

		}
		else if(e.getSource()==info)
		{
			if(oddclicks)
			{
				oddclicks=false;
				int totaleats=0;
				String [] columns={"Animal","Color","Size","Hor. speed","Ver. speed","Eat counter"};
				String[][] rows=new String[Sar.size()+1][6];
				for(int i=0;i<Sar.size();i++)
				{
					rows[i][0]=Sar.get(i).getAnimalName();
					Color c=Sar.get(i).getColor();
					if(c==Color.blue)
					{
						rows[i][1]="Blue";
					}
					else if(c==Color.red)
					{
						rows[i][1]="Red";
					}
					else if(c==Color.yellow)
					{
						rows[i][1]="Yellow";
					}
					else if(c==Color.magenta)
					{
						rows[i][1]="Magenta";
					}
					else if(c==Color.black)
					{
						rows[i][1]="Black";
					}
					else
					{
						String color="("+c.getRed()+","+c.getGreen()+","+c.getBlue()+")";
						rows[i][1]=color;
					}
					rows[i][2]=""+Sar.get(i).getSize();
					rows[i][3]=""+Sar.get(i).getHorSpeed();
					rows[i][4]=""+Sar.get(i).getVerSpeed();
					rows[i][5]=""+Sar.get(i).getEatCount();
					totaleats+=Sar.get(i).getEatCount();
				}
				String totaleat=""+totaleats;
				rows[Sar.size()][0]="Total";
				rows[Sar.size()][1]=rows[Sar.size()][2]=rows[Sar.size()][3]=rows[Sar.size()][4]="";
				rows[Sar.size()][5]=totaleat;
				JTable jt=new JTable(rows,columns);
				js=new JScrollPane(jt);
				this.add(js);
				this.setVisible(false);
				js.setVisible(true);
				this.setVisible(true);
				repaint();
			}
			else
			{
				if(js!=null)
				{
					js.setVisible(false);
					this.remove(js);
					js=null;
					oddclicks=true;
					repaint();
					
				}
			}
		}
		else if(e.getSource()==reset)
		{
			drawfood=false;
			Sar.clear();
			numofSwimmables=0;
			numofImmobiles=0;
			Par.clear();
			af.ResetSavedStates();
			food.setBackground(UIManager.getColor ( "Panel.background" ));
		}
		else if(e.getSource()==food)
		{
			food.setBackground(UIManager.getColor ( "Panel.background" ));
			int sizeofhungry=0;
			int[] arr=new int[Sar.size()];
			for(int i=0;i<Sar.size();i++)
			{
				if(Sar.get(i).isHungry())
				{
					sizeofhungry++;
					arr[i]=1;
				}
				else
					arr[i]=0;
			}
			if(sizeofhungry>0)
			{
				int indexoflasthugry=Sar.size()-1;
				while(indexoflasthugry>0)
				{
					if(arr[indexoflasthugry]==1)
						break;
					else
						indexoflasthugry--;
				}
				CyclicBarrier barrier = new CyclicBarrier(sizeofhungry);
				for(int i=0;i<indexoflasthugry;i++)
				{
					if(Sar.get(i).isHungry())
						Sar.get(i).setBarrier(barrier);
				}
				drawfood=true;
				repaint();
				Sar.get(indexoflasthugry).setBarrier(barrier);
			}
			else
				JOptionPane.showMessageDialog(null, "There is not swimmable object at the aquarium or no one is hungry currently,need at least 1!");
		}
		
	}
	/**
	 * 
	 * @param Swimmable s which is the fish or jellyfish to add to the aquarium
	 * @see AddAnimalDialog to see how the swimmable is built and sent here
	 */
	public void addSwimmable(SeaCreature s)
	{
		((Swimmable)s).setAquaPanel(this);
		((Swimmable)s).start();
		Sar.add(Sar.size(),(Swimmable)s);
		numofSwimmables+=1;
	}
	/**
	 * 
	 * @param Immobile s which is the plants to add to the aquarium
	 * @see AddAnimalDialog to see how the Immobile is built and sent here
	 */
	public void addImmobile(SeaCreature i)
	{
		((Immobile)i).setAquaPanel(this);
		Par.add(Par.size(),(Immobile)i);
		numofImmobiles+=1;
		repaint();
	}
	/**
	 * function that made to stop the cyclicbarrier
	 */
	public void StopBarrier()
	{	
		for(int i=0;i<Sar.size();i++)
		{
			Sar.get(i).StopBarrier();
		}
	}
	/**
	 * overriding of paintComponent in jpanel, to draw the animals and also the image and food
	 * @param Graphics g
	 */
	public void paintComponent(Graphics g)
	{
		super.paintComponent(g);
		if(image!=null)
			g.drawImage(image, 0,0,getWidth(),getHeight(),null);
		for(int i=0;i<Par.size();i++)
		{
			Par.get(i).drawCreature(g);
		}
		for(int i=0;i<Sar.size();i++)
		{
			Sar.get(i).drawCreature(g);
		}
		if(drawfood)
		{
			Worm worm=Worm.CreateWorm(Color.red);
			worm.drawWorm(g, getWidth(), getHeight());
		}
	}
	/**
	 * function that return the swimmable arr
	 * @return ArrayList<Swimmable>
	 */
	public ArrayList<Swimmable> getSwimmableArray()
	{
		return Sar;
	}
	/**
	 * function that return the Immobile arr
	 * @return ArrayList<Immobile>
	 */
	public ArrayList<Immobile> getImmobileArray()
	{
		return Par;
	}
	/**
	 * static function that checks if the string can be convert to int
	 * @param str that represent a string
	 * @return true if the str can be converted,else false
	 */
	public static boolean isInteger(String str) {
	    if (str == null) {
	        return false;
	    }
	    int length = str.length();
	    if (length == 0) {
	        return false;
	    }
	    int i = 0;
	    if (str.charAt(0) == '-') {
	        if (length == 1) {
	            return false;
	        }
	        i = 1;
	    }
	    for (; i < length; i++) {
	        char c = str.charAt(i);
	        if (c < '0' || c > '9') {
	            return false;
	        }
	    }
	    return true;
	}
	@Override
	/**
	 * update function that is implementation of observer interface
	 * when the swimmable is hungry, the food button becomes red
	 */
	public void update() 
	{
		this.food.setBackground(Color.red);
		
	}
}
