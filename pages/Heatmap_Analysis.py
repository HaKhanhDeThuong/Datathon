import streamlit as st 
st.set_page_config(page_title="Heatmap Analysis", page_icon="images/logo.png",layout="wide")
def title(url):
    st.markdown(f'<p style="font-size:50px; padding: 5px; font-weight: bold; text-align: center; color:#289df2">{url}</p>', unsafe_allow_html=True)
def normalText(url):
    st.markdown(f'<p style="font-size:20px; padding: 5px;  text-align: center; color:#ffffff">{url}</p>', unsafe_allow_html=True)
class Heatmap:
    def __init__(self):
        self.heatmap_images = {
            "2023-12-16": "images/heatmap1.png",
            "2023-12-17": "images/heatmap2.png",
            # Add more date-image pairs as needed
        }

    def main(self):
        title("Heatmap Analysis")
        normalText("Heatmaps visualize and analyze customer foot traffic patterns, offering insights for optimizing store layouts and enhancing overall shopping experiences.")

        st.write("---")

        selected_date = st.date_input("Choose a day to see heatmap analysis:")

        if selected_date:
            selected_date_str = str(selected_date)
            if selected_date_str in self.heatmap_images:
                heatmap_image_path = self.heatmap_images[selected_date_str]
                st.image(heatmap_image_path, width=800, caption=f"Heatmap for {selected_date_str}")
            else:
                st.warning("No heatmap available for the selected date.")
        else:
            st.warning("Please choose a date.")


if __name__ == "__main__":
   app = Heatmap()
   app.main()
 