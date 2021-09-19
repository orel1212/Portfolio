import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.FlowLayout;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;

import javax.swing.JButton;
import javax.swing.JColorChooser;
import javax.swing.JDialog;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.JTextArea;
import javax.swing.JTextField;

/**
 * JPanelDecorator class that represent the panel of the decorator
 * @author oReL
 *
 */
public class JPanelDecorator extends JPanel implements ActionListener
{
	private JButton changecolor;
	private JScrollPane js;
	private JTextField indexfield;
	private ArrayList<Swimmable> arr;
	private JPanel changepanel;
	private AquaFrame af;
	private AquaPanel ap;
	/**
	 * c'tor to build the panel
	 * @param ar-swimmable at the aquarium
	 * @param af-aquaframe 
	 * @param ap-aquapanel
	 */
	public JPanelDecorator(ArrayList<Swimmable> ar,AquaFrame af,AquaPanel ap)
	{
		super();
		this.af=af;
		this.ap=ap;
		this.setSize(700, 700);
		this.setLayout(new BorderLayout());
		arr=ar;
		changepanel=new JPanel();
		changepanel.setLayout(new FlowLayout());
		changecolor=new JButton("Change");
		changecolor.addActionListener(this);
		JLabel jl=new JLabel("Enter the index:");
		indexfield=new JTextField("",20);
		String [] columns={"Index","Animal","Color","Size","Hor. speed","Ver. speed"};
		String[][] rows=new String[ar.size()][6];
		for(int i=0;i<ar.size();i++)
		{
			rows[i][0]=""+i;
			rows[i][1]=ar.get(i).getAnimalName();
			Color c=ar.get(i).getColor();
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
			rows[i][3]=""+ar.get(i).getSize();
			rows[i][4]=""+ar.get(i).getHorSpeed();
			rows[i][5]=""+ar.get(i).getVerSpeed();
			
		}
		JTable jt=new JTable(rows,columns);
		
		js=new JScrollPane(jt);
		
		this.add(js);
		this.setVisible(false);
		js.setVisible(true);
		//this.setVisible(true);
		repaint();
		changepanel.add(jl);
		changepanel.add(indexfield);
		changepanel.add(changecolor);
		this.add(changepanel,BorderLayout.SOUTH);
		this.setVisible(true);
	}
	
	@Override
	public void actionPerformed(ActionEvent e) 
	{
		if(e.getSource()==changecolor)
		{
			if(AquaPanel.isInteger(indexfield.getText()))
			{
				int index=Integer.parseInt(indexfield.getText());
				if(index<arr.size() && index>=0)
				{
					Color c=JColorChooser.showDialog(null, "Choose your color:", Color.black);
					MarineAnimalDecorator mad=new MarineAnimalDecorator(arr.get(index), c);
					mad.PaintFish();
					af.remove(this);
					this.setVisible(false);
					af.add(ap);
					ap.setVisible(true);
				}
				else
				{
					JOptionPane.showMessageDialog(null ,"You insert index that not in the range!","Error",JOptionPane.ERROR_MESSAGE);
					indexfield.setText("");
				}
			}
			else
			{
				JOptionPane.showMessageDialog(null,"You insert some string that not represent int!", "Error",JOptionPane.ERROR_MESSAGE);
				indexfield.setText("");
			}
		}
		
	}

}
