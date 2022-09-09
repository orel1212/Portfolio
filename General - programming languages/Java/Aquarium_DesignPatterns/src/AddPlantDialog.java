
import java.awt.Color;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.Random;

import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JDialog;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JSlider;

/**
 * AddAnimalDialog which is built to make a new immobile
 * @author oReL
 */
public class AddPlantDialog extends JDialog implements ActionListener
{ 
	private JComboBox<String> cbplantes;
	private JComboBox<String> cbcolors;
	private JButton submit;
	private JSlider size;
	private static String optiondeciced;
	private AquaPanel ap;
	/**
	 * c'tor which is make a new dialog to build the immobile
	 
	 * @param a as a callback from aquapanel
	 */
	public AddPlantDialog(AquaPanel a)
	{
		super();
		ap=a;
		optiondeciced="Laminaria";
		String[] options={"Laminaria","Zostera"};
		cbplantes=new JComboBox<String>(options);
		cbplantes.addActionListener(this);
		size = new JSlider(JSlider.HORIZONTAL, 20, 320, 170);
		size.setMinorTickSpacing(0);
		size.setMajorTickSpacing(60);
		size.setPaintTicks(true);
		size.setPaintLabels(true);
		submit=new JButton("Submit");
		submit.addActionListener(this);
		this.setLayout(new GridLayout(3,2));
		this.add(new JLabel("Select Plant:"));
		this.add(cbplantes);
		this.add(new JLabel("Select Size:"));
		this.add(size);
		this.add(new JLabel("To Finish click:"));
		this.add(submit);
		this.pack();
		this.setVisible(true);
	}
	/**
	 * function that make the new immobile after the info chosen in the dailog before
	 */
	public void MakeDialog()
	{
		int sized=size.getValue();
		Color color=Color.green;
		int x= (int)(100 + (Math.random() * (ap.getWidth() - 100)));
		int y=(int)(100 + (Math.random() * (ap.getHeight() - 100)));
		PlantFactory pf=new PlantFactory(color,x,y,sized);
		if(AddPlantDialog.optiondeciced.equals("Laminaria"))
		{
			ap.addImmobile(pf.produceSeaCreature("Laminaria"));
		}
			
		else
		{
			ap.addImmobile(pf.produceSeaCreature("Zostera"));
		}
			
		dispose();
		
	}
	/**
	 * actionperformed from actionlistener interface,to implement all the functions of the dailog when clicking on the buttons/comboboxes
	 *@para Actionevent e
	 *@see MakeDialog()
	 */
	@Override
	public void actionPerformed(ActionEvent e) 
	{
		if(e.getSource()==cbplantes)
		{
			AddPlantDialog.optiondeciced=(String)cbplantes.getSelectedItem();
		}
		else if(e.getSource()==submit)
		{
			MakeDialog();
		}
		
	}
	
}
