import streamlit as st
import pandas as pd
import numpy as np
import datetime

st.set_page_config(page_title="Window Display", page_icon="images/logo.png", layout="wide")

def title(url):
    st.markdown(f'<p style="font-size:50px; padding: 5px; font-weight: bold; text-align: center; color:#289df2">{url}</p>', unsafe_allow_html=True)

def normalText(url):
    st.markdown(f'<p style="font-size:20px; padding: 5px; text-align: center; color:#ffffff">{url}</p>', unsafe_allow_html=True)

class WindowDisPlay:
    def __init__(self):
        self.advertising_banners = {
            "Banner A": "images/pic1.jpg",
            "Banner B": "images/pic2.jpg",
            "Banner C": "images/pic1.jpg",
            "Banner D": "images/pic1.jpg"
        }
        self.advertisements = {
            "Ad1": {"attention_duration": 20, "interaction_rate": 0.1, "conversion_rate_to_store_entry": 0.05},
            "Ad2": {"attention_duration": 25, "interaction_rate": 0.15, "conversion_rate_to_store_entry": 0.08},
            "Ad3": {"attention_duration": 18, "interaction_rate": 0.08, "conversion_rate_to_store_entry": 0.03},
        }

    def generate_report(self):
        # Simulate monthly data
        month = datetime.date.today().strftime("%B %Y")
        report_data = {
            "Ad Name": [],
            "Attention Duration": [],
            "Interaction Rate": [],
            "Conversion Rate to Store Entry": [],
        }

        for ad_name, ad_data in self.advertisements.items():
            report_data["Ad Name"].append(ad_name)
            report_data["Attention Duration"].append(ad_data["attention_duration"])
            report_data["Interaction Rate"].append(ad_data["interaction_rate"])
            report_data["Conversion Rate to Store Entry"].append(ad_data["conversion_rate_to_store_entry"])

        report_df = pd.DataFrame(report_data)
        return report_df, month

    def automatic_ad_selection(self):
        # Implement your logic for automatic advertisement selection based on real-time data
        # ...

        # Return the selected advertisement name (e.g., "Banner A")
        return "Banner A"

    def main(self):
        title("Window Display Controller")
        normalText("Enhance storefront appeal by dynamically adjusting window displays based on real-time video analysis of customer traffic.")
        st.write("---")
        st.markdown("Data chart about the number of customers that reach the store ")

        # Replace this line with your actual data and chart logic
        chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["col1", "col2", "col3"])

        st.line_chart(
            chart_data, x="col1", y=["col2", "col3"], color=["#FF0000", "#0000FF"]  # Optional
        )
        st.markdown("List of areas that can be useful when displaying discounts")
        st.title("Select Advertisement Banner")

        # Select an advertising banner either manually or automatically
        selected_banner = st.selectbox("Select Advertisement Banner:", list(self.advertising_banners.keys()))

        # Display the image of the selected advertising banner
        st.image(self.advertising_banners[selected_banner], width=300, caption=f"Advertisement Banner: {selected_banner}", use_column_width=True)

        st.write(f"You have selected: {selected_banner}")

        # Uncomment the following lines to enable automatic ad selection
        # automatic_selected_banner = self.automatic_ad_selection()
        # st.write(f"Automatic Advertisement Selection: {automatic_selected_banner}")

        st.title("Advertisement Performance Report")

        # Generate a monthly report
        report_df, month = self.generate_report()

        # Display the report
        st.header(f"Performance Report - {month}")
        st.dataframe(report_df, height=400)

        # Display charts (you can customize these based on your requirements)
        st.subheader("Performance Charts")

        # Add custom charts for Attention Duration, Interaction Rate, and Conversion Rate to Store Entry
        # These are placeholders; replace them with your actual logic and data
        st.bar_chart(report_df.set_index("Ad Name")[["Attention Duration", "Interaction Rate", "Conversion Rate to Store Entry"]])

        # Display Attention Duration, Interaction Rate, and Conversion Rate to Store Entry
        st.subheader("Metrics")
        st.write(f"Attention Duration: {report_df['Attention Duration'].mean()} seconds")
        st.write(f"Interaction Rate: {report_df['Interaction Rate'].mean() * 100}%")
        st.write(f"Conversion Rate to Store Entry: {report_df['Conversion Rate to Store Entry'].mean() * 100}%")

if __name__ == "__main__":
    app = WindowDisPlay()
    app.main()
